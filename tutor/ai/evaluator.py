"""Evaluation system: analiza cada turno del usuario EN PARALELO a la conversación (nunca bloquea la respuesta).

- Principal: LLM con salida estructurada (TurnEvaluation).
- Respaldo: reglas locales para errores muy típicos de hispanohablantes (si la API falla o en modo mock).
"""

from __future__ import annotations

import difflib
import logging
import re

from ..learning.curriculum import GRAMMAR
from ..learning.planner import LessonPlan
from ..learning.route import ROUTE_BY_KEY, attempts_structure, error_is_unit_related
from .llm import LLM
from .prompts import EVALUATOR_SYSTEM, evaluator_prompt
from .schemas import DetectedError, ErrorList, StructureUse, TurnEvaluation, VocabUse

log = logging.getLogger(__name__)

# (regex, corrección (puede usar grupos), categoría, skill, severidad, explicación)
_RULES: list[tuple[str, str, str, str, str, str]] = [
    (r"\b(i|you|we|they) have (went|did|saw|ate|wrote|took|came|ran)\b", r"\1 have {pp}", "verb_tense", "present_perfect", "major",
     "Con have se usa el participio: gone, done, seen…"),
    (r"\b(he|she|it) has (went|did|saw|ate|wrote|took|came|ran)\b", r"\1 has {pp}", "verb_tense", "present_perfect", "major",
     "Con has se usa el participio: gone, done, seen…"),
    (r"\byesterday,? (i|we|they|he|she) (go|have|make|talk|do|see|eat|work|meet|say|tell|buy)\b", r"yesterday \1 {past}",
     "verb_tense", "past_simple", "major", "Para algo que ya pasó, usamos el pasado."),
    (r"\b(last \w+),? (i|we|they|he|she) (go|have|make|talk|do|see|eat|work|meet|say|tell|buy)\b", r"\1 \2 {past}",
     "verb_tense", "past_simple", "major", "Para algo que ya pasó, usamos el pasado."),
    (r"\bi am agree\b", "I agree", "grammar", "verb_patterns", "major", "Agree es un verbo: I agree, sin am."),
    (r"\bpeople is\b", "people are", "grammar", "present_simple", "minor", "People es plural: people are."),
    (r"\b(he|she|it|my (?:wife|husband|boss|manager|brother|sister|mother|mum|mom|father|dad|friend|son|daughter|"
     r"partner|colleague|team|company|family)) don'?t\b", r"\1 doesn't", "grammar", "present_simple", "major", "Con he/she/it: doesn't."),
    (r"\bexplain me\b", "explain to me", "preposition", "prepositions", "minor", "Se dice explain to me / explain it to me."),
    (r"\bdepends? of\b", "depends on", "preposition", "prepositions", "minor", "Depend va con on."),
    (r"\b(made|make|making) a meeting\b", "had a meeting", "naturalness", "word_choice", "minor", "Con meeting se usa have, no make."),
    (r"\bi have (\d+|twenty|thirty|forty|fifty) years\b", r"I'm \1 years old", "grammar", "present_simple", "major",
     "La edad se dice con be: I'm 30 (years old)."),
    (r"\bsince (\d+|two|three|four|five|six|ten) (years|months)\b", r"for \1 \2", "preposition", "present_perfect", "major",
     "Para duraciones usamos for, no since."),
    (r"\bmore (better|bigger|easier|worse)\b", r"\1", "grammar", "comparatives", "major", "Better ya es comparativo: sin more."),
    (r"\bthe (life|people|money) is\b", r"\1 is", "article", "articles", "minor", "En general, sin the: life is…, money is…"),
    # he/she sin -s ("my sister work"), salvo detrás de un auxiliar ("does she work", "can he go", "let her go")
    (r"(?<!does )(?<!did )(?<!can )(?<!could )(?<!will )(?<!would )(?<!should )(?<!must )(?<!might )(?<!to )"
     r"(?<!let )(?<!make )(?<!help )(?<!doesn't )(?<!didn't )(?<!don't )"
     r"\b(he|she|it|my (?:wife|husband|boss|manager|brother|sister|mother|mum|mom|father|dad|friend|son|daughter|"
     r"partner|colleague|team|company|family))"
     r" (?:(?:always|usually|often|sometimes|never|also|really|still) )?"
     r"(work|live|go|like|want|need|start|finish|get|take|make|play|watch|study|drive|teach|help|eat|speak|know|"
     r"think|come|have|do|love|leave|walk|cook|wake|travel|spend|prefer|enjoy|use|try|deal|manage|buy|check|fix|call|"
     r"run|sell|clean|open|close|pay|write|send|answer|ask|look|feel|see|say|tell|bring|wear|sleep|listen|stay|move|"
     r"change|wash|catch|miss|commute|organise|organize|prepare|visit|drink|mean|seem|hate|stop|forget|remember|"
     r"understand|believe|plan|hope|learn|meet|keep|give|show|lose|win)\b",
     "{s3}", "grammar", "present_simple", "major", "Con he/she, el verbo en presente lleva -s."),
    # negación a la española ("I no like", "my boss no work")
    (r"\b(i|you|we|they) no (work|like|have|go|know|want|need|eat|drink|live|understand|play|speak|drive|watch|use)\b",
     r"\1 don't \2", "grammar", "present_simple", "major", "Para negar en presente: don't + verbo."),
    (r"\b(he|she|it|my \w+) no (work|like|have|go|know|want|need|eat|drink|live|understand|play|speak|drive|watch|use)\b",
     r"\1 doesn't \2", "grammar", "present_simple", "major", "Con he/she, para negar: doesn't + verbo."),
    # adverbio de frecuencia detrás del verbo ("I go always")
    (r"\b(i|we|they|you) (?!am\b|are\b|was\b|were\b|'m\b|'re\b)(go|work|eat|play|watch|read|cook|get|do|have|take|visit|call|see|drink|drive|walk) (always|usually|often|sometimes|never)\b",
     r"\1 \3 \2", "word_order", "frequency_adverbs", "major", "Always, usually, often… van antes del verbo."),
]


def third_person(verb: str) -> str:
    if verb == "have":
        return "has"
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"
_PP = {"went": "gone", "did": "done", "saw": "seen", "ate": "eaten", "wrote": "written", "took": "taken", "came": "come", "ran": "run"}
_PAST = {"go": "went", "have": "had", "make": "made", "talk": "talked", "do": "did", "see": "saw", "eat": "ate",
         "work": "worked", "meet": "met", "say": "said", "tell": "told", "buy": "bought"}


def rule_based_evaluation(text: str, plan: LessonPlan | None = None) -> TurnEvaluation:
    low = text.lower()
    errors: list[DetectedError] = []
    for pattern, repl, cat, skill, sev, expl in _RULES:
        for m in re.finditer(pattern, low):
            corrected = m.expand(repl)
            if "{pp}" in corrected:
                corrected = corrected.replace("{pp}", _PP.get(m.group(2), m.group(2)))
            if "{s3}" in corrected:
                verb = m.groups()[-1]
                corrected = m.group(0)[: m.end(m.lastindex) - m.start()].rsplit(" ", 1)[0] + " " + third_person(verb)
            if "{past}" in corrected:
                verb = m.groups()[-1]
                corrected = corrected.replace("{past}", _PAST.get(verb, verb + "ed"))
            errors.append(DetectedError(original=m.group(0), corrected=corrected, category=cat, skill_key=skill,
                                        severity=sev, explanation_es=expl))
    used = []
    if plan:
        for term in plan.all_target_terms:
            if term_used(term, text):
                used.append(VocabUse(term=term, correct=True))
    structures = [StructureUse(skill_key=e.skill_key, correct=False) for e in errors if e.skill_key in GRAMMAR]
    if re.search(r"\b(went|had|was|were|did|made|talked|worked|saw|met)\b", low) and not any(
            e.skill_key == "past_simple" for e in errors):
        structures.append(StructureUse(skill_key="past_simple", correct=True))
    if re.search(r"\b(have|has|'ve) (been|gone|done|seen|worked|had|made)\b", low) and not any(
            e.skill_key == "present_perfect" for e in errors):
        structures.append(StructureUse(skill_key="present_perfect", correct=True))
    unit = ROUTE_BY_KEY.get((plan.unit or {}).get("key", "")) if plan else None
    if unit and attempts_structure(unit, text) and not any(s.skill_key == unit.skill for s in structures):
        structures.append(StructureUse(skill_key=unit.skill, correct=True))
    words = len(low.split())
    complexity = 1 if words < 4 else 2 if words < 10 else 3 if words < 25 else 4
    return TurnEvaluation(errors=errors, target_vocab_used=used, structures_used=structures, complexity=complexity,
                          understood_question=True)


def term_used(term: str, text: str) -> bool:
    """¿Aparece el término (con flexión simple) en el texto? 'to postpone' ~ 'postponed'."""
    words = re.sub(r"^to ", "", term.lower()).split()
    if not words:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(w[:-1] if len(w) > 4 and w.endswith("e") else w) + r"\w{0,4}" for w in words) + r"\b"
    return re.search(pattern, text.lower()) is not None


def _overlaps(a: str, b: str) -> bool:
    a, b = a.lower().strip(" .,"), b.lower().strip(" .,")
    return a in b or b in a


_PRONOUNS = {"i", "you", "we", "they", "he", "she", "it", "my", "our", "your", "his", "her", "their"}
_CONTRACTIONS = {"i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are", "he's": "he is",
                 "she's": "she is", "it's": "it is", "that's": "that is", "there's": "there is", "don't": "do not",
                 "doesn't": "does not", "didn't": "did not", "isn't": "is not", "aren't": "are not",
                 "wasn't": "was not", "weren't": "were not", "can't": "cannot", "won't": "will not",
                 "i've": "i have", "we've": "we have", "they've": "they have", "you've": "you have",
                 "i'll": "i will", "i'd": "i would", "haven't": "have not", "hasn't": "has not",
                 "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not"}
_HELP_REQUEST = re.compile(
    r"\b(i (don't|do not|didn't|can't|cannot) (understand|follow)|(can|could) you (repeat|say (it|that) again)|"
    r"repeat,? please|(say|speak) (more )?(slowly|slower)|what do you mean|sorry,? what|pardon|i don't get it|"
    r"no (te )?entiendo|no he entendido|puedes repetir|otra vez)\b", re.I)


_WORD_REQUEST = re.compile(
    r"\b(?:how (?:do you|do i|can i|could i|would you|to|should i) say|(?:i )?(?:don't|do not) know how to say|"
    r"what(?:'s| is) the word for|what(?:'s| is) (?:the english for|it in english)|"
    r"c[oó]mo se dice|no s[eé] c[oó]mo (?:se )?dec(?:ir|e))\s+[\"“'‘]?([^,.?!\"”'’]{2,40}?)[\"”'’]?"
    r"(?:\s+(?:in english|en ingl[eé]s))?\s*(?=[,.?!]|$|\s+(?:and|but|so|because)\b)", re.I)


def word_request(text: str) -> str:
    """Palabra o expresión que el alumno pide en inglés ("I don't know how to say lejos" → "lejos"), o ""."""
    m = _WORD_REQUEST.search(text.replace("’", "'"))
    if not m:
        return ""
    word = m.group(1).strip()
    if word.lower() in ("it", "that", "this", "the word", "this word", "that word"):
        return word  # no dijo qué palabra: el profesor le pregunta cuál
    return word


def asks_for_help(text: str) -> bool:
    """¿El alumno dice que no ha entendido o pide que se repita?"""
    return bool(_HELP_REQUEST.search(text.replace("’", "'")))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def _expanded(text: str) -> list[str]:
    return " ".join(_CONTRACTIONS.get(w, w) for w in _words(text)).split()


_TOKEN_STRIP = " ,.;:!?\"'“”‘’()"
CONTEXT_BEFORE, CONTEXT_AFTER = 2, 1  # palabras sin cambios alrededor del cambio, para que se entienda dónde está
MAX_CHANGED_WORDS = 6  # un cambio más largo es reescribir la frase, no un error concreto
MAX_CHANGES = 3


def _norm_token(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", "", tok.lower().replace("’", "'"))


def narrow_error(e: DetectedError) -> list[DetectedError]:
    """Acota un error al fragmento que realmente cambia (más un par de palabras de contexto). Los modelos pequeños
    suelen dar la frase entera como "original"; así el alumno ve exactamente qué palabra está mal. Si la corrección
    reescribe la frase (muchos cambios o muy largos) no es un error concreto y se descarta."""
    o = [t for t in e.original.split() if _norm_token(t)]
    c = [t for t in e.corrected.split() if _norm_token(t)]
    no, nc = [_norm_token(t) for t in o], [_norm_token(t) for t in c]
    ops = [op for op in difflib.SequenceMatcher(None, no, nc, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return [e]  # sin cambios reales: plausible_error lo descartará
    groups: list[list[tuple]] = [[ops[0]]]
    for op in ops[1:]:
        if op[1] - groups[-1][-1][2] <= 2:  # cambios separados por 1-2 palabras: un solo error
            groups[-1].append(op)
        else:
            groups.append([op])
    changed_total = sum(max(op[2] - op[1], op[4] - op[3]) for op in ops)
    if len(groups) > MAX_CHANGES or (len(o) >= 8 and changed_total * 2 > len(o)):
        return []  # reescribe la frase: no es un error concreto (la respuesta modelo ya enseña cómo decirlo)
    out = []
    for g in groups:
        changed = max(g[-1][2] - g[0][1], g[-1][4] - g[0][3])
        if changed > MAX_CHANGED_WORDS:
            continue
        before, after = CONTEXT_BEFORE, CONTEXT_AFTER
        # el contexto no cruza una coma o un punto: "up, I get a" → "I get a"
        for k in range(1, CONTEXT_BEFORE + 1):
            if g[0][1] - k < 0 or re.search(r"[,.;:!?]$", o[g[0][1] - k]):
                before = k - 1
                break
        last = o[g[-1][2] - 1] if g[-1][2] > g[-1][1] else ""
        if re.search(r"[,.;:!?]$", last):
            after = 0
        i1, i2 = max(0, g[0][1] - before), min(len(o), g[-1][2] + after)
        j1, j2 = max(0, g[0][3] - before), min(len(c), g[-1][4] + after)
        orig = " ".join(o[i1:i2]).strip(_TOKEN_STRIP)
        corr = " ".join(c[j1:j2]).strip(_TOKEN_STRIP)
        if not (orig and corr):
            continue
        err = e.model_copy(update={"original": orig, "corrected": corr})
        if len(groups) > 1:
            # Una explicación para varios cambios sólo vale para el cambio del que habla.
            touched = {w for op in g for w in no[op[1]:op[2]] + nc[op[3]:op[4]] if len(w) > 2}
            expl = e.explanation_es.lower().replace("’", "'")
            if not any(re.search(rf"\b{re.escape(w)}\b", expl) for w in touched):
                err = err.model_copy(update={"explanation_es": ""})
        out.append(err)
    return out


_SPANISH = re.compile(r"\b(el|la|los|las|que|qué|es|se|para|con|una?|por|del|de|en|y|tu|su|al|cuando|como|verbo|frase|más|"
                      r"usa(mos)?|lleva|dice|significa|sin|mejor|aquí|después|antes)\b", re.I)


def is_spanish(text: str) -> bool:
    return len(_SPANISH.findall(text)) >= 2


def fix_explanation(e: DetectedError) -> DetectedError:
    """La explicación debe estar en español; si el modelo la escribió en inglés se sustituye por una genérica."""
    if is_spanish(e.explanation_es):
        return e
    generic = {"naturalness": "Así lo diría un nativo; es más natural.", "word_choice": "Esta palabra es la que se usa en este caso.",
               "preposition": "Esta es la preposición que se usa aquí.", "article": "Aquí hace falta (o sobra) el artículo.",
               "word_order": "En inglés el orden de las palabras es este.", "verb_tense": "Este es el tiempo verbal que pide la frase."}
    return e.model_copy(update={"explanation_es": generic.get(e.category, "Así es como se dice correctamente en inglés.")})


_LEAK = re.compile(r"[#·→{}<>]|”,|\bskill_key\b|\bexplanation\b", re.I)


def plausible_error(e: DetectedError, learner_text: str, unclear: list[str] | tuple = ()) -> bool:
    """Descarta "errores" inventados (frecuentes en modelos pequeños): el fragmento debe aparecer en lo que dijo el
    alumno, la corrección debe cambiar algo (expandir "I'm" a "I am" no cuenta), no vale cambiar sólo el sujeto o
    un pronombre, y no se juzgan palabras que el reconocedor de voz no entendió bien (no son errores del alumno)."""
    if "*" in e.corrected or "*" in e.original or _LEAK.search(e.corrected) or _LEAK.search(e.original):
        return False  # texto del prompt o de código colado en la corrección
    orig, corr, said = _words(e.original), _words(e.corrected), " ".join(_words(learner_text))
    if not orig or orig == corr or " ".join(orig) not in said:
        return False
    if _expanded(e.original) == _expanded(e.corrected):
        return False
    dubious = {w for u in unclear for w in _words(u)}
    if dubious & set(orig):
        return False
    # La corrección debe ser la misma frase arreglada, no otra frase ("Yes, I enjoyed it" → "I'm starting a new job").
    if difflib.SequenceMatcher(None, _expanded(e.original), _expanded(e.corrected)).ratio() < 0.5:
        return False
    if len(orig) == len(corr):
        diffs = [(a, b) for a, b in zip(orig, corr) if a != b]
        if diffs and all(a in _PRONOUNS and b in _PRONOUNS for a, b in diffs):
            return False
    # Cambiar de persona ("I live" → "he lives") cambia lo que quiso decir; he ↔ she sí puede ser un error real.
    persons = {"i": 1, "we": 1, "you": 2, "he": 3, "she": 3, "it": 3, "they": 3}
    o_p = [persons[w] for w in orig if w in persons]
    c_p = [persons[w] for w in corr if w in persons]
    if o_p and c_p and len(o_p) == len(c_p) and o_p != c_p:
        return False
    return True


def merge_evaluations(llm: TurnEvaluation, rules: TurnEvaluation, learner_text: str = "",
                      unclear: list[str] | tuple = ()) -> TurnEvaluation:
    """Errores de las reglas locales (precisos y seguros) + errores del LLM acotados al fragmento que cambia,
    verificados y que no se solapen con los de las reglas (sin duplicados)."""
    errors: list[DetectedError] = list(rules.errors)
    for raw in llm.errors:
        if _LEAK.search(raw.corrected) or _LEAK.search(raw.original):
            continue  # texto del prompt colado: el modelo se ha liado con todo el error
        for e in narrow_error(raw):
            if learner_text and not plausible_error(e, learner_text, unclear):
                continue
            if any(_overlaps(e.original, x.original) for x in errors):
                continue  # ya lo marca una regla, o el modelo local repite el mismo error
            errors.append(fix_explanation(e))
    vocab = list(llm.target_vocab_used)
    for v in rules.target_vocab_used:
        if not any(v.term.lower() == x.term.lower() for x in vocab):
            vocab.append(v)
    structures = list(llm.structures_used) or list(rules.structures_used)
    return llm.model_copy(update={"errors": errors, "target_vocab_used": vocab, "structures_used": structures})


def upgrade_unit_errors(ev: TurnEvaluation, plan: LessonPlan | None) -> TurnEvaluation:
    """Un error con la estructura de la clase de hoy siempre es importante: es lo que se está aprendiendo."""
    unit = ROUTE_BY_KEY.get((plan.unit or {}).get("key", "")) if plan else None
    if not unit:
        return ev
    errors = [e.model_copy(update={"severity": "major"}) if error_is_unit_related(unit, e.model_dump()) else e
              for e in ev.errors]
    return ev.model_copy(update={"errors": errors})


class Evaluator:
    """`errors_only=True` (modelos locales): el LLM sólo busca errores (verificados después); estructuras,
    vocabulario y complejidad se calculan con reglas, porque un modelo pequeño no los juzga de forma fiable."""

    def __init__(self, llm: LLM | None, errors_only: bool = False):
        self.llm = llm
        self.errors_only = errors_only

    async def evaluate(self, tutor_text: str, learner_text: str, plan: LessonPlan, known_patterns: list[dict],
                       unclear: list[str] | tuple = ()) -> TurnEvaluation:
        if asks_for_help(learner_text):
            # Pedir ayuda ("I don't understand", "Can you repeat?") no se corrige: sólo cuenta como no entendido.
            ev = rule_based_evaluation(learner_text, plan) if len(learner_text.split()) <= 10 else \
                await self._evaluate(tutor_text, learner_text, plan, known_patterns, unclear)
            return ev.model_copy(update={"understood_question": False})
        return await self._evaluate(tutor_text, learner_text, plan, known_patterns, unclear)

    async def _evaluate(self, tutor_text: str, learner_text: str, plan: LessonPlan, known_patterns: list[dict],
                        unclear: list[str] | tuple) -> TurnEvaluation:
        if self.llm is None:
            return rule_based_evaluation(learner_text, plan)
        rules = rule_based_evaluation(learner_text, plan)
        if self.errors_only:
            try:
                found = await self.llm.parse(EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)),
                                             evaluator_prompt(tutor_text, learner_text, plan, known_patterns), ErrorList)
            except Exception as e:  # noqa: BLE001
                log.warning("Evaluación local fallida (%s); usando reglas", e)
                return rules
            recurring = {m["original"].lower() for m in known_patterns}
            # Un modelo pequeño marca casi todo como "major"; una palabra poco natural no lo es salvo que sea un
            # error que el alumno ya repite.
            found_errors = [e.model_copy(update={"severity": "minor"})
                            if e.category in ("word_choice", "naturalness", "vocabulary")
                            and e.original.lower() not in recurring else e for e in found.errors]
            llm_part = TurnEvaluation(errors=found_errors, complexity=rules.complexity, understood_question=True)
            merged = upgrade_unit_errors(merge_evaluations(llm_part, rules, learner_text, unclear), plan)
            extra = [StructureUse(skill_key=e.skill_key, correct=False) for e in merged.errors
                     if e.skill_key in GRAMMAR and not any(s.skill_key == e.skill_key for s in rules.structures_used)]
            structures = [s for s in rules.structures_used
                          if not (s.correct and any(e.skill_key == s.skill_key for e in merged.errors))] + extra
            unique = list({(s.skill_key, s.correct): s for s in structures}.values())
            return merged.model_copy(update={"structures_used": unique})
        try:
            ev = await self.llm.parse(EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)),
                                      evaluator_prompt(tutor_text, learner_text, plan, known_patterns), TurnEvaluation)
        except Exception as e:  # la evaluación nunca debe romper la lección
            log.warning("Evaluación LLM fallida (%s); usando reglas locales", e)
            return rules
        return upgrade_unit_errors(merge_evaluations(ev, rules, learner_text, unclear), plan)
