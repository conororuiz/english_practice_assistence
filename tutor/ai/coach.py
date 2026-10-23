"""Análisis de cada respuesta del alumno para la pantalla: qué ha comunicado y cómo decirlo mejor, con el porqué,
cómo se usa y un ejemplo nuevo de cada sugerencia.

Va después de la evaluación (en segundo plano, nunca retrasa al tutor) y parte de sus errores ya verificados: el
modelo sólo los explica, no puede añadir correcciones propias (un modelo pequeño inventaría errores). Si el modelo
falla o no hay modelo, las sugerencias se construyen con la explicación de cada error y la regla de su unidad.
"""

from __future__ import annotations

import logging
import re

from ..learning.planner import LessonPlan
from ..learning.route import ROUTE
from .evaluator import _overlaps, _words, plausible_error
from .llm import LLM
from .prompts import COACH_SYSTEM, coach_prompt
from .schemas import DetectedError, Suggestion, TurnFeedback

log = logging.getLogger(__name__)

MAX_SUGGESTIONS = 3
_SPANISH = re.compile(r"\b(el|la|los|las|que|qué|es|se|para|con|una?|por|del|de|en|y|tu|su|al|cuando|como|verbo|frase|más|usa(mos)?)\b",
                      re.I)


def worth_coaching(text: str) -> bool:
    return len(_words(text)) >= 2


def is_spanish(text: str) -> bool:
    return len(_SPANISH.findall(text)) >= 2


def _unit_example(unit, avoid: str) -> str:
    return next((x for x in unit.examples if _words(x) != _words(avoid)), "")


def unit_for(e: dict):
    """Unidad de la ruta que explica este error ("made a party" → "had a party" no es una cuestión de pasado
    simple, aunque las dos frases estén en pasado)."""
    # La categoría y la destreza que pone el evaluador no son fiables; la señal buena es la estructura que aparece al
    # corregir y no estaba en lo que dijo el alumno.
    uses = [u for u in ROUTE if re.search(u.detect, e["corrected"], re.I) and not re.search(u.detect, e["original"], re.I)]
    return next((u for u in uses if u.skill == e["skill_key"]), uses[0] if uses else None)


def _from_error(e: dict) -> Suggestion:
    """Sugerencia sin modelo: la explicación del error y, si la hay, la regla y un ejemplo de su unidad."""
    unit = unit_for(e)
    example = _unit_example(unit, e["corrected"]) if unit else ""
    return Suggestion(kind="correction", you_said=e["original"], better=e["corrected"], why_es=e["explanation_es"],
                      how_es=f"{unit.explain_es} ({unit.form})" if unit else "", example=example)


def rule_based_feedback(errors: list[dict]) -> TurnFeedback:
    suggestions = [_from_error(e) for e in sorted(errors, key=lambda e: e["severity"] != "major")][:MAX_SUGGESTIONS]
    if not errors:
        analysis = "Respuesta correcta: se entiende bien lo que quieres decir."
    elif any(e["severity"] == "major" for e in errors):
        analysis = "Se entiende tu idea, pero hay un error importante que conviene corregir."
    else:
        analysis = "Se entiende bien; sólo hay un detalle pequeño para sonar más natural."
    return TurnFeedback(analysis_es=analysis, suggestions=suggestions, model_answer="", model_answer_why_es="",
                        model_answer_how_es="")


def _matching_error(s: Suggestion, errors: list[dict]) -> dict | None:
    """El error verificado al que se refiere una corrección (el modelo a veces deja you_said vacío)."""
    for e in errors:
        if (s.you_said.strip() and _overlaps(s.you_said, e["original"])) or _overlaps(s.better, e["corrected"]):
            return e
    return None


def _explained_correction(s: Suggestion, e: dict) -> Suggestion:
    """Corrección anclada a su error verificado. Si el error es de una estructura de la ruta, la regla y el ejemplo
    salen de la unidad (revisados a mano), no del modelo: un modelo pequeño explica mal la gramática."""
    base = _from_error(e)
    unit = unit_for(e)
    if unit:
        return base.model_copy(update={"why_es": s.why_es if is_spanish(s.why_es) else base.why_es})
    if not (is_spanish(s.why_es) and is_spanish(s.how_es)):
        return base
    return s.model_copy(update={"kind": "correction", "you_said": e["original"], "better": e["corrected"]})


def clean_feedback(fb: TurnFeedback, learner_text: str, errors: list[dict]) -> TurnFeedback:
    """Descarta lo que un modelo pequeño se inventa: correcciones que no son de la evaluación, fragmentos que el
    alumno no dijo, explicaciones vacías o que no están en español. Los errores importantes verificados que el
    modelo no explicó se añaden igualmente."""
    kept: list[Suggestion] = []
    loose: list[Suggestion] = []  # sugerencias explicadas que no se pueden comprobar
    for s in fb.suggestions:
        if not s.better.strip() or s.kind == "richer_answer":
            continue  # la respuesta modelo va en su propio campo
        e = _matching_error(s, errors)  # los modelos a veces dan un error verificado como "more_natural"
        if e is not None:
            if any(k.kind == "correction" and _overlaps(k.you_said, e["original"]) for k in kept):
                continue
            s = _explained_correction(s, e)
        elif s.kind == "correction":
            continue
        elif not (is_spanish(s.why_es) and is_spanish(s.how_es)):
            continue
        else:
            probe = DetectedError(original=s.you_said, corrected=s.better, category="naturalness",
                                  skill_key="word_choice", severity="minor", explanation_es=s.why_es)
            if not plausible_error(probe, learner_text):
                loose.append(s)
                continue
        if s.example.strip().lower() == s.better.strip().lower():
            s = s.model_copy(update={"example": ""})
        if any(_words(s.better) == _words(k.better) for k in kept) or (s.kind != "correction" and any(
                _overlaps(s.better, e["corrected"]) for e in errors)):
            continue
        kept.append(s)
    # Todo error verificado debe aparecer, aunque el modelo lo haya olvidado.
    missing = [e for e in sorted(errors, key=lambda e: e["severity"] == "major")
               if not any(k.kind == "correction" and _overlaps(k.you_said, e["original"]) for k in kept)]
    if len(missing) == 1 and len(loose) == 1 and not loose[0].you_said.strip():
        # Un solo error sin explicar y una sola explicación suelta (sin fragmento): es la de ese error con otras palabras.
        kept.insert(0, _explained_correction(loose[0].model_copy(update={"example": loose[0].example}), missing[0]))
        missing = []
    for e in missing:
        kept.insert(0, _from_error(e))
    kept.sort(key=lambda s: s.kind != "correction")
    kept = kept[:MAX_SUGGESTIONS]
    # Respuesta modelo: sólo si dice más que el alumno y viene explicada en español.
    model = fb.model_answer.strip()
    if (len(_words(model)) > len(_words(learner_text)) and is_spanish(fb.model_answer_why_es)
            and is_spanish(fb.model_answer_how_es)):
        kept.append(Suggestion(kind="richer_answer", better=model, why_es=fb.model_answer_why_es,
                               how_es=fb.model_answer_how_es, example=""))
    analysis = fb.analysis_es.strip()
    if not is_spanish(analysis):
        analysis = rule_based_feedback(errors).analysis_es
    return fb.model_copy(update={"analysis_es": analysis, "suggestions": kept})


class Coach:
    def __init__(self, llm: LLM | None):
        self.llm = llm

    async def feedback(self, tutor_text: str, learner_text: str, errors: list[dict], plan: LessonPlan) -> TurnFeedback:
        if self.llm is None:
            return rule_based_feedback(errors)
        try:
            fb = await self.llm.parse(COACH_SYSTEM, coach_prompt(tutor_text, learner_text, errors, plan), TurnFeedback)
        except Exception as e:  # noqa: BLE001 — el análisis nunca debe romper la lección
            log.warning("Análisis de la respuesta fallido (%s); usando los errores verificados", e)
            return rule_based_feedback(errors)
        return clean_feedback(fb, learner_text, errors)
