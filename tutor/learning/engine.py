"""Learning engine: convierte lo ocurrido en una lección en progreso persistente.

Entrada: turnos del usuario (texto + métricas de audio + evaluación), eventos de vocabulario
registrados en vivo y el plan con el que se ejecutó la lección.
Salida: métricas de la sesión, perfil actualizado (gramática, vocabulario, errores, nivel, paciencia)
y los datos para el feedback final.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .curriculum import GRAMMAR
from .fluency import fluency_index, session_fluency
from .memory import UserMemory
from .planner import LessonPlan


@dataclass
class LiveVocabLog:
    """Eventos de vocabulario detectados durante la lección (en orden)."""

    events: list[tuple[str, str, str | None]] = field(default_factory=list)  # (term, event, meaning_es)
    introduced: dict[str, str] = field(default_factory=dict)
    elicited_pending: dict[str, int] = field(default_factory=dict)  # term -> nº de turno en que se provocó
    used: set[str] = field(default_factory=set)

    def introduce(self, term: str, meaning_es: str | None) -> None:
        term = term.lower().strip()
        if term not in self.introduced:
            self.introduced[term] = meaning_es or ""
            self.events.append((term, "introduced", meaning_es))
        else:
            self.events.append((term, "exposed", None))

    def elicit(self, term: str, turn_idx: int) -> None:
        term = term.lower().strip()
        self.elicited_pending[term] = turn_idx
        self.events.append((term, "elicited", None))

    def use(self, term: str, turn_idx: int, tutor_recent_text: str, correct: bool = True) -> str:
        """Clasifica un uso del usuario como provocado o espontáneo. Devuelve el evento registrado."""
        term = term.lower().strip()
        if not correct:
            ev = "failed"
        else:
            elicited_at = self.elicited_pending.pop(term, None)
            prompted = (elicited_at is not None and turn_idx - elicited_at <= 2) or term in tutor_recent_text.lower()
            ev = "used_prompted" if prompted else "used_spontaneous"
        self.used.add(term)
        self.events.append((term, ev, None))
        return ev

    def close(self) -> None:
        """Lo que se intentó provocar y nunca salió cuenta como fallo (vuelve pronto a repaso)."""
        for term in list(self.elicited_pending):
            if term not in self.used:
                self.events.append((term, "failed", None))
        self.elicited_pending.clear()


def _compare(current: float, previous: list[float]) -> str:
    if not previous:
        return "first"
    base = statistics.mean(previous)
    if current >= base + 3:
        return "better"
    if current <= base - 3:
        return "lower"
    return "similar"


class LearningEngine:
    def __init__(self, memory: UserMemory):
        self.memory = memory

    def finalize(self, session_id: int, plan: LessonPlan, user_turns: list[dict], vocab: LiveVocabLog,
                 lesson_seconds: float, now: float | None = None) -> dict:
        """`user_turns`: [{turn_id, text, audio: dict, eval: dict | None}]. Devuelve las métricas de la sesión."""
        now = now or time.time()
        mem = self.memory
        vocab.close()

        # 1. Errores y dominio gramatical
        per_skill: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "major": 0, "minor": 0})
        session_errors: list[dict] = []
        complexities: list[int] = []
        comprehension_issues = 0
        good_vocab: list[str] = []
        for t in user_turns:
            ev = t.get("eval")
            if not ev:
                continue
            complexities.append(ev.get("complexity", 3))
            if not ev.get("understood_question", True):
                comprehension_issues += 1
            good_vocab.extend(ev.get("good_vocabulary", []))
            if ev.get("likely_asr_issue"):
                continue
            for e in ev.get("errors", []):
                mem.add_error(session_id, t.get("turn_id"), e, now)
                session_errors.append(e)
                per_skill[e["skill_key"]][e["severity"]] += 1
            for s in ev.get("structures_used", []):
                if s.get("correct"):
                    per_skill[s["skill_key"]]["correct"] += 1

        improvements, skill_changes = [], {}
        for key, c in per_skill.items():
            before, after = mem.update_skill(key, c["correct"], c["major"], c["minor"], now)
            skill_changes[key] = {"before": before, "after": after, **c}
            if before < 0.7 <= after and c["correct"] >= 2:
                improvements.append(GRAMMAR.get(key, {"name": key})["name"])

        # 2. Vocabulario (repetición espaciada)
        domain = plan.domain
        for term, event, meaning in vocab.events:
            mem.vocab_event(session_id, term, event, meaning_es=meaning, domain=domain, now=now)

        # 3. Fluidez
        fluency = session_fluency(user_turns, lesson_seconds)
        fi = fluency_index(fluency)
        words = max(fluency["total_words"], 1)
        majors = sum(1 for e in session_errors if e["severity"] == "major")
        minors = len(session_errors) - majors
        previous = [s["metrics"]["fluency_index"] for s in mem.db.recent_sessions(mem.user_id, 3)
                    if s.get("metrics") and "fluency_index" in s["metrics"]]

        # Errores repetidos dentro de la misma sesión
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for e in session_errors:
            counts[(e["original"].lower(), e["corrected"])] += 1

        metrics = {
            "fluency": fluency,
            "fluency_index": fi,
            "fluency_trend": _compare(fi, previous),
            "errors": {
                "major": majors,
                "minor": minors,
                "major_per_100w": round(100 * majors / words, 2),
                "minor_per_100w": round(100 * minors / words, 2),
                "by_skill": {k: {"major": v["major"], "minor": v["minor"], "correct": v["correct"]} for k, v in per_skill.items()},
                "repeated": [{"original": o, "corrected": c, "count": n} for (o, c), n in counts.items() if n >= 2],
                "list": session_errors,
            },
            "complexity_mean": round(statistics.mean(complexities), 2) if complexities else None,
            "comprehension_issues": comprehension_issues,
            "vocab": {
                "introduced": [{"term": t, "meaning_es": es} for t, es in vocab.introduced.items()],
                "used": sorted(vocab.used),
                "events": [{"term": t, "event": e} for t, e, _ in vocab.events if e != "exposed"],
            },
            "good_vocabulary": sorted(set(good_vocab))[:10],
            "skill_changes": skill_changes,
            "improvements": improvements,
        }

        # 4. Nivel estimado (conservador: media móvil lenta, empieza en B1 como indicó el usuario)
        if fluency["turns"] >= 3:
            complexity = metrics["complexity_mean"] or 3
            perf = 0.5 * fi / 100 + 0.3 * (1 - min(metrics["errors"]["major_per_100w"] / 6, 1)) + 0.2 * (complexity - 1) / 4
            observed = max(0.0, min(4.0, (perf - 0.45) / 0.15))
            _, score = mem.level()
            mem.set_level(0.85 * score + 0.15 * observed)
        metrics["level"] = mem.level()[0]

        # 5. Perfil: mejoras recientes (el planner las usa para reducir correcciones de ese tema)
        profile = mem.profile()
        profile["recent_improvements"] = improvements
        mem.save_profile(profile)
        return metrics

    def apply_summary_profile(self, strengths: list[str], weaknesses: list[str]) -> None:
        profile = self.memory.profile()
        profile["strengths"] = strengths[:4]
        profile["weaknesses"] = weaknesses[:4]
        self.memory.save_profile(profile)
