"""Feedback de fin de sesión: los datos los calcula el learning engine; el LLM sólo los redacta.
Si el LLM falla, se genera un resumen determinista con los mismos datos."""

from __future__ import annotations

import logging

from ..learning.planner import LessonPlan
from .llm import LLM
from .prompts import SUMMARY_SYSTEM, summary_prompt
from .schemas import SessionSummary, SummaryVocab

log = logging.getLogger(__name__)

_TREND = {
    "first": "First lesson done — great start! 🎉",
    "better": "Better than last time — more fluent and confident.",
    "similar": "Similar to your last lessons — steady progress.",
    "lower": "A harder day today — totally normal, keep going.",
}


def build_summary_input(metrics: dict, plan: LessonPlan, next_plan: LessonPlan, snapshot: dict,
                        corrections_given: list[dict]) -> dict:
    errs = metrics["errors"]
    return {
        "fluency_trend_vs_previous_lessons": metrics["fluency_trend"],
        "fluency_metrics": {k: metrics["fluency"][k] for k in (
            "speaking_wpm", "mean_turn_words", "median_onset_s", "long_pauses_per_min", "fillers_per_100w", "stuck_events")},
        "major_errors": [e for e in errs["list"] if e["severity"] == "major"][:8],
        "minor_errors": [e for e in errs["list"] if e["severity"] == "minor"][:6],
        "repeated_errors_this_lesson": errs["repeated"],
        "corrections_given_aloud": corrections_given,
        "new_vocabulary_introduced": [
            {"term": v["term"], "meaning_es": v["meaning_es"] or next(
                (p["meaning_es"] for p in plan.new_vocab if p["term"].lower() == v["term"]), "")}
            for v in metrics["vocab"]["introduced"]],
        "target_vocabulary_used_by_learner": metrics["vocab"]["used"],
        "good_vocabulary_used": metrics["good_vocabulary"],
        "grammar_improvements": metrics["improvements"],
        "overused_words": metrics["fluency"]["overused_words"],
        "today_topic": plan.title,
        "minutes": max(1, round(metrics.get("duration_s", 900) / 60)),
        "turns": metrics["fluency"]["turns"],
        "next_lesson": {"topic": next_plan.title, "grammar": [g["name"] for g in next_plan.target_grammar],
                        "review_words": [v["term"] for v in next_plan.review_vocab],
                        "same_unit_as_today": (next_plan.unit or {}).get("key") == (plan.unit or {}).get("key"),
                        "class_in_unit": next_plan.unit_lesson},
        "unit_progress": metrics.get("unit"),
        "previous_strengths": snapshot.get("strengths", []),
        "previous_weaknesses": snapshot.get("weaknesses", []),
    }


def deterministic_summary(data: dict, plan: LessonPlan) -> SessionSummary:
    majors = data["major_errors"]
    improve = [f"{e['skill_key'].replace('_', ' ').capitalize()}: \"{e['original']}\" → \"{e['corrected']}\"" for e in majors[:2]]
    if not improve and data["minor_errors"]:
        e = data["minor_errors"][0]
        improve.append(f"\"{e['original']}\" → \"{e['corrected']}\"")
    if data["fluency_metrics"]["long_pauses_per_min"] > 2:
        improve.append("Long pauses: try short linking phrases like 'Let me think…' or 'The thing is…'")
    vocab = [SummaryVocab(term=v["term"], meaning_es=v["meaning_es"]) for v in data["new_vocabulary_introduced"]]
    fm = data["fluency_metrics"]
    well_done = []
    if data["target_vocabulary_used_by_learner"]:
        well_done.append("You used: " + ", ".join(f"'{t}'" for t in data["target_vocabulary_used_by_learner"][:3]) + " 👏")
    if fm["mean_turn_words"] >= 15:
        well_done.append(f"Long, detailed answers ({fm['mean_turn_words']:.0f} words on average).")
    if fm.get("median_onset_s") is not None and fm["median_onset_s"] <= 2.0:
        well_done.append("You answered quickly, without long hesitations.")
    if fm["stuck_events"] == 0 and fm["long_pauses_per_min"] < 1:
        well_done.append("Few long pauses — good flow.")
    if not well_done:
        well_done.append(f"You kept the class going for {data.get('minutes', 15)} minutes.")
    pool = data["new_vocabulary_introduced"] + plan.new_vocab
    expr = next((v for v in pool if " " in v["term"] and v.get("meaning_es")),
                next((v for v in pool if v.get("meaning_es")), {"term": "Let me think about that…", "meaning_es": "Déjame pensarlo…"}))
    nl = data["next_lesson"]
    if nl.get("same_unit_as_today"):
        tomorrow = f"We'll keep practising {nl['topic']} (class {nl['class_in_unit']}) until you master it."
    elif plan.unit:
        tomorrow = f"New unit: {nl['topic']}. 🎉"
    else:
        tomorrow = f"We'll practise {nl['topic'].lower()}" + (f" and {nl['grammar'][0].lower()}" if nl["grammar"] else "") + "."
    return SessionSummary(
        fluency_line=_TREND.get(data["fluency_trend_vs_previous_lessons"], ""),
        new_vocabulary=vocab,
        improve=improve or ["Keep going — no important errors today! 👏"],
        well_done=well_done[:2],
        useful_expression=expr["term"],
        useful_expression_es=expr["meaning_es"],
        tomorrow=tomorrow,
        strengths=data["previous_strengths"],
        weaknesses=sorted({e["skill_key"].replace("_", " ") for e in majors})[:4] or data["previous_weaknesses"],
    )


class Summarizer:
    def __init__(self, llm: LLM | None, effort: str = "medium"):
        self.llm = llm
        self.effort = effort

    async def summarize(self, data: dict, plan: LessonPlan) -> SessionSummary:
        if self.llm is None:
            return deterministic_summary(data, plan)
        try:
            return await self.llm.parse(SUMMARY_SYSTEM, summary_prompt(data), SessionSummary, effort=self.effort)
        except Exception as e:
            log.warning("Resumen LLM fallido (%s); usando resumen determinista", e)
            return deterministic_summary(data, plan)
