"""Lesson planner: decide qué se practica en la próxima lección a partir del perfil del usuario.

Estructura general (arco semanal) + adaptación: escenario, gramática a reforzar (dentro de la
conversación, nunca como clase), vocabulario a reciclar según la repetición espaciada, errores
recurrentes y notas de adaptación ("has mejorado X → menos correcciones, más fluidez").
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

from .curriculum import GRAMMAR, SCENARIOS, WEEKLY_ARC, Scenario
from .memory import UserMemory

SESSIONS_PER_WEEK = 5


@dataclass
class LessonPlan:
    lesson_number: int
    week: int
    weekly_focus: str
    weekly_goal: str
    scenario_key: str
    domain: str
    title: str
    situation: str
    roleplay: bool
    target_grammar: list[dict] = field(default_factory=list)  # [{key, name, elicit, mastery}]
    review_vocab: list[dict] = field(default_factory=list)  # [{term, meaning_es, stage}]
    new_vocab: list[dict] = field(default_factory=list)  # [{term, meaning_es}]
    recurring_mistakes: list[dict] = field(default_factory=list)
    adaptation_notes: list[str] = field(default_factory=list)
    correction_intensity: str = "normal"  # light | normal

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LessonPlan":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def all_target_terms(self) -> list[str]:
        return [v["term"] for v in self.review_vocab] + [v["term"] for v in self.new_vocab]


def _target_difficulty(level_score: float) -> int:
    return 1 if level_score < 1.0 else 2 if level_score < 2.0 else 3


def plan_next_lesson(memory: UserMemory, now: float | None = None) -> LessonPlan:
    now = now or time.time()
    completed = memory.db.count_completed_sessions(memory.user_id)
    lesson_number = completed + 1
    week = completed // SESSIONS_PER_WEEK
    recent = memory.db.recent_sessions(memory.user_id, limit=6)
    last_metrics = (recent[0].get("metrics") or {}) if recent else {}
    fluency = last_metrics.get("fluency", {})

    # --- foco semanal (estructura general) con ajuste al rendimiento real ---
    arc = WEEKLY_ARC[min(week, len(WEEKLY_ARC) - 1)]
    notes: list[str] = []
    if arc["focus"] in ("spontaneity", "professional", "complex_ideas") and fluency and fluency.get("mean_turn_words", 99) < 8:
        arc = WEEKLY_ARC[2]  # todavía respuestas muy cortas: seguimos trabajando respuestas largas
        notes.append("Answers are still short: keep pushing for longer answers (reasons, examples).")

    # --- dominio: alternar cotidiano / trabajo, sesgado por foco y debilidades ---
    last_domain = recent[0]["plan"]["domain"] if recent and recent[0].get("plan") else "work"
    domain = "everyday" if last_domain == "work" else "work"
    if arc["focus"] == "professional" and lesson_number % 3 != 0:
        domain = "work"

    # --- gramática a reforzar ---
    weak = memory.weak_skills(now, limit=3)
    weak_keys = [w["skill_key"] for w in weak]

    # --- escenario ---
    level_name, level_score = memory.level()
    target = _target_difficulty(level_score)
    recent_keys = {s["plan"]["scenario_key"] for s in recent[:4] if s.get("plan")}
    rng = random.Random(lesson_number * 7919 + memory.user_id)

    def score(s: Scenario) -> float:
        sc = 0.0
        sc += 3.0 if s.domain == domain else 0.0
        sc -= 1.5 * abs(s.difficulty - target)
        sc += 2.0 * len(set(s.grammar_fit) & set(weak_keys))
        sc -= 5.0 if s.key in recent_keys else 0.0
        if arc["focus"] == "professional" and s.roleplay:
            sc += 1.0
        return sc + rng.random() * 0.5

    scenario = SCENARIOS[0] if lesson_number == 1 else max(SCENARIOS, key=score)

    target_grammar = []
    for w in weak[:2]:
        target_grammar.append({"key": w["skill_key"], "name": w["name"], "elicit": w["elicit"], "mastery": round(w["mastery"], 2)})
    if not target_grammar:  # sin datos todavía: diagnóstico con lo que el escenario hace aparecer
        for key in scenario.grammar_fit[:1]:
            g = GRAMMAR[key]
            target_grammar.append({"key": key, "name": g["name"], "elicit": g["elicit"], "mastery": None})

    # --- vocabulario: repaso espaciado + nuevo del escenario ---
    review = []
    for item in memory.due_vocab(now, limit=4):
        box = item["box"]
        stage = "reuse" if box == 0 else "elicit" if box <= 2 else "check_spontaneous"
        review.append({"term": item["term"], "meaning_es": item.get("meaning_es") or "", "stage": stage})
    known = memory.known_terms()
    n_new = 3 if len(review) <= 1 else 2 if len(review) <= 3 else 1
    if arc["focus"] == "vocabulary":
        n_new += 1
    new_vocab = [{"term": t, "meaning_es": es} for t, es in scenario.seed_vocab if t.lower() not in known][:n_new]

    # --- adaptación explícita: lo que ha mejorado deja de ser foco ---
    profile = memory.profile()
    for imp in profile.get("recent_improvements", []):
        notes.append(f"The learner has improved a lot with {imp}: correct it less and spend more time on fluency and vocabulary. "
                     f"Tell them this briefly and positively at the start.")
    intensity = "normal"
    if fluency and (fluency.get("stuck_events", 0) >= 3 or fluency.get("long_pauses_per_min", 0) > 2.5):
        intensity = "light"
        notes.append("The learner struggled with fluency last time: keep corrections to a minimum, scaffold more, and build confidence.")

    return LessonPlan(
        lesson_number=lesson_number,
        week=week + 1,
        weekly_focus=arc["focus"],
        weekly_goal=arc["goal"],
        scenario_key=scenario.key,
        domain=scenario.domain,
        title=scenario.title,
        situation=scenario.situation,
        roleplay=scenario.roleplay,
        target_grammar=target_grammar,
        review_vocab=review,
        new_vocab=new_vocab,
        recurring_mistakes=[{"original": m["original"], "corrected": m["corrected"], "count": m["count"]}
                            for m in memory.recurring_mistakes(limit=3)],
        adaptation_notes=notes,
        correction_intensity=intensity,
    )
