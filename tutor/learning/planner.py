"""Lesson planner: decide la próxima clase a partir de la ruta de aprendizaje y del perfil del usuario.

La clase es la de la unidad actual de la ruta (la primera sin dominar). Dentro de la unidad, cada clase
cambia: vocabulario nuevo hasta presentar las 8 palabras, después se insiste en las que aún no ha usado;
ejercicios que todavía no ha resuelto bien primero; otro contexto para la conversación; y un foco explícito
en lo que le falta para dominarla. Además se recicla: vocabulario de repetición espaciada, errores
recurrentes, una estructura débil de antes y, de vez en cuando, una pregunta de repaso de una unidad dominada.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

from .curriculum import GRAMMAR, SCENARIOS_BY_KEY
from .memory import UserMemory
from .route import ROUTE, ROUTE_BY_KEY, TALK, unit_index, unit_status
from .srs import DAY

SESSIONS_PER_WEEK = 5
NEW_WORDS_PER_LESSON = 4


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
    # --- ruta de aprendizaje ---
    unit: dict = field(default_factory=dict)  # Unit como dict (key, title, explain_es, form, examples, watch…)
    unit_number: int = 1  # posición en la ruta (1..N)
    unit_count: int = len(ROUTE)
    unit_lesson: int = 1  # nº de clase dentro de la unidad
    unit_status: dict = field(default_factory=dict)  # dominio antes de esta clase (score, missing…)
    drills: list[str] = field(default_factory=list)  # ejercicios guiados, en orden
    focus: list[str] = field(default_factory=list)  # lo que hay que trabajar hoy para dominar la unidad
    review_unit: dict | None = None  # repaso rápido de una unidad dominada: {title, form, drill}
    just_mastered: str | None = None  # título de la unidad recién dominada (para felicitar al empezar)
    production_scenario: str = ""  # contexto concreto de la conversación libre
    talk: list[str] = field(default_factory=list)  # preguntas para la conversación real, en orden

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LessonPlan":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def all_target_terms(self) -> list[str]:
        return [v["term"] for v in self.review_vocab] + [v["term"] for v in self.new_vocab]

    @property
    def unit_terms(self) -> list[str]:
        return [t for t, _ in self.unit.get("vocab", [])]


def _unit_dict(u) -> dict:
    d = asdict(u)
    d["vocab"] = [list(v) for v in u.vocab]
    d["watch"] = [list(w) for w in u.watch]
    return d


def plan_next_lesson(memory: UserMemory, now: float | None = None) -> LessonPlan:
    now = now or time.time()
    completed = memory.db.count_completed_sessions(memory.user_id)
    lesson_number = completed + 1
    recent = memory.db.recent_sessions(memory.user_id, limit=6)
    last_metrics = (recent[0].get("metrics") or {}) if recent else {}
    fluency = last_metrics.get("fluency", {})
    rng = random.Random(lesson_number * 7919 + memory.user_id)

    # --- unidad actual de la ruta ---
    unit = ROUTE_BY_KEY[memory.current_unit_key()]
    progress = memory.unit_progress(unit.key) or {"lessons": 0, "correct": 0, "major": 0, "minor": 0, "drills_done": []}
    unit_lesson = progress["lessons"] + 1
    words = memory.unit_words_state([t for t, _ in unit.vocab])
    status = unit_status(progress, sum(1 for s in words.values() if s == "used"))
    last_plan = recent[0].get("plan") if recent else None
    just_mastered = None
    if last_plan and (last_plan.get("unit") or {}).get("key") not in (None, unit.key):
        just_mastered = last_plan["unit"]["title"]

    # --- contexto de la conversación: cambia en cada clase de la unidad ---
    scenario = SCENARIOS_BY_KEY[unit.scenarios[(unit_lesson - 1) % len(unit.scenarios)]]

    # --- vocabulario: palabras nuevas de la unidad, luego las presentadas que aún no ha usado ---
    meanings = {t.lower(): es for t, es in unit.vocab}
    unseen = [t for t, s in words.items() if s == "unseen"]
    pending = [t for t, s in words.items() if s == "introduced"]
    new_vocab = [{"term": t, "meaning_es": meanings[t]} for t in unseen[:NEW_WORDS_PER_LESSON]]
    review = [{"term": t, "meaning_es": meanings[t], "stage": "elicit"} for t in pending[:4]]
    unit_terms = set(meanings)
    for item in memory.due_vocab(now, limit=6):  # repetición espaciada de unidades anteriores
        if item["term"] not in unit_terms and len(review) < 6:
            box = item["box"]
            stage = "reuse" if box == 0 else "elicit" if box <= 2 else "check_spontaneous"
            review.append({"term": item["term"], "meaning_es": item.get("meaning_es") or "", "stage": stage})

    # --- ejercicios: primero los que aún no ha resuelto bien, en un orden distinto cada clase ---
    done = set(progress.get("drills_done") or [])
    fresh = [d for d in unit.drills if d not in done]
    old = [d for d in unit.drills if d in done]
    rng.shuffle(old)
    shift = (unit_lesson - 1) * 3 % max(len(fresh), 1)
    drills = fresh[shift:] + fresh[:shift] + old

    # --- gramática: la estructura de la unidad + una estructura débil de antes ---
    skills = memory.skills()
    g = GRAMMAR[unit.skill]
    target_grammar = [{"key": unit.skill, "name": g["name"], "elicit": g["elicit"],
                       "mastery": round(skills[unit.skill]["mastery"], 2) if unit.skill in skills else None}]
    studied = {u.skill for u in ROUTE[:unit_index(unit.key)]}
    for w in memory.weak_skills(now, limit=4):
        if w["skill_key"] != unit.skill and w["skill_key"] in studied:
            target_grammar.append({"key": w["skill_key"], "name": w["name"], "elicit": w["elicit"],
                                   "mastery": round(w["mastery"], 2)})
            break

    # --- repaso de una unidad ya dominada (la que lleva más tiempo sin repasar) ---
    review_unit = None
    mastered = [(k, p) for k, p in memory.all_unit_progress().items() if p["status"] == "mastered"]
    if mastered:
        k, p = min(mastered, key=lambda kp: kp[1].get("last_review_at") or kp[1].get("mastered_at") or 0)
        if now - (p.get("last_review_at") or p.get("mastered_at") or 0) >= 2 * DAY:
            ru = ROUTE_BY_KEY[k]
            review_unit = {"key": k, "title": ru.title, "form": ru.form, "drill": rng.choice(ru.drills)}

    # --- foco de hoy: lo que falta para dominar la unidad ---
    focus: list[str] = []
    if unit_lesson > 1:
        acc = status.get("last_accuracy")
        if acc is not None and acc < 0.8:
            focus.append(f"Accuracy with {unit.title.split(':')[0]} was {round(acc * 100)}% last time: practise it "
                         "carefully and correct every mistake with it.")
        if pending:
            focus.append("Words introduced but not used yet by the learner: " + ", ".join(pending[:4]))
        errs = [m for m in memory.recurring_mistakes(limit=5) if m.get("skill_key") == unit.skill]
        for m in errs[:2]:
            focus.append(f"Recurring mistake with this structure: \"{m['original']}\" → \"{m['corrected']}\"")

    # --- adaptación ---
    notes: list[str] = []
    profile = memory.profile()
    for imp in profile.get("recent_improvements", []):
        notes.append(f"The learner has improved a lot with {imp}: correct it less and tell them briefly and "
                     "positively at the start.")
    intensity = "normal"
    if fluency and (fluency.get("stuck_events", 0) >= 3 or fluency.get("long_pauses_per_min", 0) > 2.5):
        intensity = "light"
        notes.append("The learner struggled with fluency last time: outside the exercises keep corrections to a "
                     "minimum, scaffold more, and build confidence.")
    if unit_lesson >= 4:
        notes.append(f"This is lesson {unit_lesson} of this unit and it is not mastered yet: explain it again in a "
                     "different, simpler way, with very easy examples and smaller steps. Be very encouraging.")

    return LessonPlan(
        lesson_number=lesson_number,
        week=completed // SESSIONS_PER_WEEK + 1,
        weekly_focus=unit.key,
        weekly_goal=unit.goal,
        scenario_key=scenario.key,
        domain=scenario.domain,
        title=unit.title,
        situation=scenario.situation,
        roleplay=scenario.roleplay,
        target_grammar=target_grammar,
        review_vocab=review,
        new_vocab=new_vocab,
        recurring_mistakes=[{"original": m["original"], "corrected": m["corrected"], "count": m["count"]}
                            for m in memory.recurring_mistakes(limit=3)],
        adaptation_notes=notes,
        correction_intensity=intensity,
        unit=_unit_dict(unit),
        unit_number=unit_index(unit.key) + 1,
        unit_lesson=unit_lesson,
        unit_status=status,
        drills=drills,
        focus=focus,
        review_unit=review_unit,
        just_mastered=just_mastered,
        production_scenario=f"{unit.production} Setting: {scenario.title}.",
        talk=_rotate(list(TALK.get(unit.key, ())), (unit_lesson - 1) * 2),
    )


def _rotate(items: list, k: int) -> list:
    if not items:
        return items
    k %= len(items)
    return items[k:] + items[:k]
