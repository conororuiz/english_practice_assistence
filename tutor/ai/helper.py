"""Ayudas durante la clase, a petición del alumno (botones de la pantalla; nunca interrumpen la clase):

  - «Ayúdame a responder»: 2-3 comienzos de frase para la pregunta del profesor y palabras que le pueden servir.
  - «Traducir»: la última frase del profesor en español (sólo en pantalla).
  - «Dila tú»: el alumno repite una frase corregida y se compara palabra a palabra con la frase objetivo.

Un modelo pequeño se equivoca: todo se valida (comienzos en inglés y cortos, traducción en español) y, si el modelo
falla, hay ayudas por reglas.
"""

from __future__ import annotations

import difflib
import logging
import re

from pydantic import BaseModel, Field

from .evaluator import _SPANISH
from .llm import LLM

log = logging.getLogger(__name__)

MAX_STARTERS = 3
MAX_WORDS = 3

HELP_SYSTEM = """You help an English learner (level B1, Spanish speaker) who doesn't know how to start answering \
their teacher's question.

Give:
- starters: 2-3 different ways to BEGIN an answer, in natural English. Each one is an unfinished sentence (max 10 \
words) ending with "…" that the learner completes with their own information. Use today's structure when it fits. \
Use what you know about the learner if it helps, but never invent facts about them.
- words: up to 3 useful English words or short expressions for this answer, each with its meaning in Spanish.

Never answer the question for them completely. Only English in starters."""

TRANSLATE_SYSTEM = """You are a translator. Translate the English text into natural Latin American Spanish, for a \
learner who didn't understand their teacher. Every sentence of your output must be in Spanish (questions too). Only \
a word in quotes that is being taught may stay in English. Output only the Spanish translation."""


class HelpWord(BaseModel):
    en: str
    es: str


class AnswerHelp(BaseModel):
    starters: list[str] = Field(default_factory=list, description="Unfinished English sentences ending with …")
    words: list[HelpWord] = Field(default_factory=list)


class Translation(BaseModel):
    es: str = Field(description="The Spanish translation")


def last_question(text: str) -> str:
    """La pregunta a la que tiene que responder el alumno: la última frase con «?» (o la última frase)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    return (questions or sentences or [""])[-1]


_YES_NO = re.compile(r"^(do|does|did|are|is|was|were|have|has|can|could|would|will)\b", re.I)


def rule_help(question: str, plan_form: str = "") -> AnswerHelp:
    """Comienzos genéricos según el tipo de pregunta (sin modelo, o si el modelo falla)."""
    q = question.strip().lower()
    if _YES_NO.match(q):
        aux = q.split()[0]
        aux = {"does": "do", "is": "am", "are": "am", "was": "was", "were": "was", "has": "have"}.get(aux, aux)
        neg = {"do": "don't", "did": "didn't", "am": "'m not", "was": "wasn't", "have": "haven't", "can": "can't",
               "could": "couldn't", "would": "wouldn't", "will": "won't"}.get(aux, "not")
        no = f"No, I{neg}" if neg.startswith("'") else f"No, I {neg}"
        starters = [f"Yes, I {aux}. For example, …", f"{no}, because …", "Sometimes. It depends on …"]
    elif q.startswith("why"):
        starters = ["Because …", "The main reason is that …", "I think it's because …"]
    elif q.startswith("where"):
        starters = ["I usually … in …", "Near my …", "It's in …"]
    elif q.startswith("when") or "what time" in q:
        starters = ["Usually at …", "Every … I …", "It depends, but normally …"]
    elif q.startswith("how"):
        starters = ["I usually … by …", "Well, first I …", "It's quite … because …"]
    else:
        starters = ["Well, I think …", "For me, …", "Let me think… I usually …"]
    return AnswerHelp(starters=starters[:MAX_STARTERS], words=[])


def _clean_starter(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().strip('"“”')).rstrip(".")
    s = re.sub(r"\s*(\.\.\.|…)+$", "", s)
    return f"{s} …" if s else ""


def _not_english(text: str) -> bool:
    return bool(_SPANISH.search(text) or re.search(r"[áéíóúñ¿¡]|\b(yo|mi|pero|normalmente)\b", text, re.I))


def clean_help(h: AnswerHelp, question: str, plan_form: str = "") -> AnswerHelp:
    starters, seen = [], set()
    for s in h.starters:
        c = _clean_starter(s)
        key = c.lower()
        if not c or key in seen or _not_english(c) or len(c.split()) > 12 or "?" in c:
            continue
        seen.add(key)
        starters.append(c)
    words = [w for w in h.words if w.en.strip() and w.es.strip() and len(w.en.split()) <= 4
             and not _not_english(w.en) and w.en.strip().lower() != w.es.strip().lower()][:MAX_WORDS]
    if len(starters) < 2:
        starters = (starters + [s for s in rule_help(question, plan_form).starters if s.lower() not in seen])
    return AnswerHelp(starters=starters[:MAX_STARTERS], words=words)


def help_prompt(question: str, context: str, unit_form: str, facts: list[str]) -> str:
    parts = [f"Teacher's question: {question}"]
    if context and context != question:
        parts.append(f"What the teacher said before it: {context[-300:]}")
    if unit_form:
        parts.append(f"Today's structure: {unit_form}")
    if facts:
        parts.append("What we know about the learner: " + "; ".join(facts[:6]))
    return "\n".join(parts)


def corrected_sentence(text: str, said: str, better: str, max_words: int = 18) -> str:
    """La frase del alumno con la corrección aplicada (para repetirla entera), o el fragmento si no se encuentra."""
    if not said.strip():
        return better
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        m = re.search(re.escape(said.strip()), sentence, re.I)
        if m:
            fixed = (sentence[:m.start()] + better.strip() + sentence[m.end():]).strip()
            fixed = fixed[:1].upper() + fixed[1:]
            return fixed if len(fixed.split()) <= max_words else better
    return better


def phrase_seen(fragment: str, text: str) -> bool:
    """El fragmento aparece en lo que dijo el alumno (mismas palabras seguidas, sin mirar mayúsculas ni signos)."""
    f, t = _norm_words(fragment), _norm_words(text)
    return bool(f) and any(t[i:i + len(f)] == f for i in range(len(t) - len(f) + 1))


# ---------------------------------------------------------------- «Dila tú»: comparar con la frase objetivo
def _norm_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")) if w]


def compare_repeat(target: str, said: str, unclear: list[str] | None = None) -> dict:
    """Qué palabras de la frase objetivo dijo el alumno. Devuelve la nota (0-100), las palabras que faltaron o
    se dijeron distinto, y las que Whisper no entendió bien (posible pronunciación)."""
    t, s = _norm_words(target), _norm_words(said)
    if not t:
        return {"score": 0, "missing": [], "extra": [], "unclear": []}
    sm = difflib.SequenceMatcher(a=t, b=s, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    missing, extra = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            missing.extend(t[i1:i2])
        if op in ("replace", "insert"):
            extra.extend(s[j1:j2])
    score = matched / len(t)
    if len(s) > len(t):  # palabras de más también restan, aunque menos
        score -= 0.5 * (len(s) - matched) / len(t)
    unclear_words = [w for w in (unclear or []) if w.lower().strip(".,!?") in t]
    return {"score": max(0, round(score * 100)), "missing": missing, "extra": extra, "unclear": unclear_words}


def repeat_reaction(score: int) -> str:
    if score >= 90:
        return "That's it, well done!"
    if score >= 70:
        return "Very close! Look at the screen to see what changed."
    return "Good try. Listen once more and try again when you want."


class Helper:
    def __init__(self, llm: LLM | None):
        self.llm = llm
        self._translations: dict[str, str] = {}

    async def help(self, tutor_text: str, unit_form: str = "", facts: list[str] | None = None) -> AnswerHelp:
        question = last_question(tutor_text)
        if self.llm is None or not question:
            return rule_help(question, unit_form)
        try:
            out = await self.llm.parse(HELP_SYSTEM, help_prompt(question, tutor_text, unit_form, facts or []),
                                       AnswerHelp)
        except Exception as e:  # noqa: BLE001 — una ayuda nunca debe romper la clase
            log.warning("Ayuda para responder fallida (%s); se usan comienzos por reglas", e)
            return rule_help(question, unit_form)
        return clean_help(out, question, unit_form)

    async def translate(self, text: str) -> str:
        """Traduce lo que el profesor dijo en inglés. Lo que ya dijo en español (<es>…</es>) no se traduce."""
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<es>.*?</es>", " ", text, flags=re.S))).strip()
        if not text:
            return ""
        if text in self._translations:
            return self._translations[text]
        if self.llm is None:
            return ""
        try:
            out = (await self.llm.parse(TRANSLATE_SYSTEM, f"English text: {text}", Translation)).es.strip()
            out = re.sub(r"\s+", " ", re.sub(r"</?es>", "", out))
        except Exception as e:  # noqa: BLE001
            log.warning("Traducción fallida (%s)", e)
            return ""
        if not (_SPANISH.search(out) or re.search(r"[áéíóúñ¿¡]", out)) or out.lower() == text.lower():
            return ""
        self._translations[text] = out
        return out
