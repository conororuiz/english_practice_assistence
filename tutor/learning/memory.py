"""User memory: operaciones de dominio sobre el progreso almacenado (vocabulario, gramática, errores, perfil)."""

from __future__ import annotations

import json
import re
import time

from ..audio.turn_detector import PatienceProfile
from .curriculum import GRAMMAR
from .db import Database
from .srs import apply_vocab_event, review_interval_days, update_mastery, DAY

LEVELS = ("B1", "B1+", "B2", "B2+", "C1")


def level_from_score(score: float) -> str:
    return LEVELS[min(int(max(score, 0.0)), len(LEVELS) - 1)]


def pattern_key(original: str, corrected: str) -> str:
    norm = lambda s: re.sub(r"[^a-z' ]+", "", s.lower()).strip()  # noqa: E731
    return f"{norm(original)} -> {norm(corrected)}"


class UserMemory:
    def __init__(self, db: Database, user_id: int = 1):
        self.db = db
        self.user_id = user_id
        db.get_or_create_user(user_id)

    # --- perfil ---------------------------------------------------------
    @property
    def user(self):
        return self.db.get_or_create_user(self.user_id)

    def profile(self) -> dict:
        return json.loads(self.user["profile_json"] or "{}")

    def save_profile(self, profile: dict) -> None:
        self.db.update_user(self.user_id, profile_json=profile)

    def patience(self) -> PatienceProfile:
        return PatienceProfile.from_dict(json.loads(self.user["patience_json"] or "{}"))

    def save_patience(self, p: PatienceProfile) -> None:
        self.db.update_user(self.user_id, patience_json=p.to_dict())

    def level(self) -> tuple[str, float]:
        u = self.user
        return u["level"], float(u["level_score"])

    def set_level(self, score: float) -> None:
        self.db.update_user(self.user_id, level_score=round(score, 3), level=level_from_score(score))

    # --- vocabulario ----------------------------------------------------
    def vocab_item(self, term: str) -> dict | None:
        row = self.db.one("SELECT * FROM vocab WHERE user_id=? AND term=?", (self.user_id, term.lower()))
        return dict(row) if row else None

    def vocab_event(self, session_id: int, term: str, event: str, meaning_es: str | None = None,
                    domain: str | None = None, kind: str | None = None, now: float | None = None) -> dict:
        now = now or time.time()
        term = term.strip().lower()
        item = self.vocab_item(term)
        if item is None:
            if event != "introduced":
                # Sólo seguimos palabras que el tutor ha presentado (evita llenar la tabla de ruido).
                return {}
            kind = kind or ("expression" if " " in term else "word")
            self.db.ex(
                "INSERT INTO vocab(user_id, term, kind, meaning_es, domain, introduced_at) VALUES(?,?,?,?,?,?)",
                (self.user_id, term, kind, meaning_es, domain, now),
            )
            item = self.vocab_item(term)
            assert item is not None
        elif meaning_es and not item.get("meaning_es"):
            item["meaning_es"] = meaning_es
        updated = apply_vocab_event(item, event, now)
        self.db.ex(
            "UPDATE vocab SET box=?, status=?, next_due=?, exposures=?, elicitations=?, prompted_uses=?, "
            "spontaneous_uses=?, failures=?, last_used=?, meaning_es=? WHERE user_id=? AND term=?",
            (updated["box"], updated["status"], updated["next_due"], updated["exposures"], updated["elicitations"],
             updated["prompted_uses"], updated["spontaneous_uses"], updated["failures"], updated["last_used"],
             updated.get("meaning_es"), self.user_id, term),
        )
        self.db.ex("INSERT INTO vocab_events(user_id, session_id, term, event, created_at) VALUES(?,?,?,?,?)",
                   (self.user_id, session_id, term, event, now))
        return updated

    def due_vocab(self, now: float, limit: int = 4) -> list[dict]:
        rows = self.db.q(
            "SELECT * FROM vocab WHERE user_id=? AND status!='mastered' AND next_due<=? ORDER BY next_due LIMIT ?",
            (self.user_id, now, limit),
        )
        return [dict(r) for r in rows]

    def known_terms(self) -> set[str]:
        return {r["term"] for r in self.db.q("SELECT term FROM vocab WHERE user_id=?", (self.user_id,))}

    def vocab_overview(self) -> dict:
        rows = [dict(r) for r in self.db.q("SELECT * FROM vocab WHERE user_id=? ORDER BY introduced_at DESC", (self.user_id,))]
        by_status: dict[str, list[dict]] = {}
        for r in rows:
            by_status.setdefault(r["status"], []).append(r)
        return {"total": len(rows), "by_status": {k: len(v) for k, v in by_status.items()}, "items": rows}

    # --- gramática ------------------------------------------------------
    def skills(self) -> dict[str, dict]:
        return {r["skill_key"]: dict(r) for r in self.db.q("SELECT * FROM skills WHERE user_id=?", (self.user_id,))}

    def update_skill(self, skill_key: str, correct: int, major: int, minor: int, now: float) -> tuple[float, float]:
        """Devuelve (dominio_anterior, dominio_nuevo)."""
        cur = self.db.one("SELECT * FROM skills WHERE user_id=? AND skill_key=?", (self.user_id, skill_key))
        before = float(cur["mastery"]) if cur else 0.5
        after = update_mastery(before, correct, major, minor)
        nxt = now + review_interval_days(after) * DAY
        if cur is None:
            self.db.ex(
                "INSERT INTO skills(user_id, skill_key, mastery, errors_total, correct_total, last_practiced, next_review) "
                "VALUES(?,?,?,?,?,?,?)", (self.user_id, skill_key, after, major + minor, correct, now, nxt))
        else:
            self.db.ex(
                "UPDATE skills SET mastery=?, errors_total=errors_total+?, correct_total=correct_total+?, "
                "last_practiced=?, next_review=? WHERE user_id=? AND skill_key=?",
                (after, major + minor, correct, now, nxt, self.user_id, skill_key))
        return before, after

    def weak_skills(self, now: float, limit: int = 3) -> list[dict]:
        rows = self.db.q(
            "SELECT * FROM skills WHERE user_id=? AND mastery<0.75 AND (next_review IS NULL OR next_review<=?) "
            "ORDER BY mastery ASC LIMIT ?", (self.user_id, now + 0.5 * DAY, limit))
        out = []
        for r in rows:
            d = dict(r)
            d.update(GRAMMAR.get(d["skill_key"], {"name": d["skill_key"], "elicit": ""}))
            out.append(d)
        return out

    # --- errores --------------------------------------------------------
    def add_error(self, session_id: int, turn_id: int | None, e: dict, now: float | None = None) -> None:
        self.db.ex(
            "INSERT INTO errors(user_id, session_id, turn_id, category, skill_key, severity, original, corrected, "
            "explanation_es, pattern_key, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (self.user_id, session_id, turn_id, e["category"], e["skill_key"], e["severity"], e["original"],
             e["corrected"], e.get("explanation_es"), pattern_key(e["original"], e["corrected"]), now or time.time()),
        )

    def recurring_mistakes(self, min_count: int = 2, limit: int = 5, since_sessions: int = 8) -> list[dict]:
        rows = self.db.q(
            """SELECT pattern_key, MAX(original) AS original, MAX(corrected) AS corrected, MAX(skill_key) AS skill_key,
                      COUNT(*) AS count, COUNT(DISTINCT session_id) AS sessions, MAX(created_at) AS last_seen
               FROM errors WHERE user_id=? AND session_id IN
                    (SELECT id FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT ?)
               GROUP BY pattern_key HAVING COUNT(*)>=? ORDER BY count DESC, last_seen DESC LIMIT ?""",
            (self.user_id, self.user_id, since_sessions, min_count, limit))
        return [dict(r) for r in rows]

    def pattern_count(self, original: str, corrected: str) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM errors WHERE user_id=? AND pattern_key=?",
                          (self.user_id, pattern_key(original, corrected)))
        return int(row["n"]) if row else 0

    # --- ruta de aprendizaje ---------------------------------------------
    def unit_progress(self, unit_key: str) -> dict | None:
        row = self.db.one("SELECT * FROM unit_progress WHERE user_id=? AND unit_key=?", (self.user_id, unit_key))
        if row is None:
            return None
        d = dict(row)
        d["drills_done"] = json.loads(d["drills_done"] or "[]")
        return d

    def all_unit_progress(self) -> dict[str, dict]:
        out = {}
        for r in self.db.q("SELECT unit_key FROM unit_progress WHERE user_id=?", (self.user_id,)):
            out[r["unit_key"]] = self.unit_progress(r["unit_key"])
        return out

    def current_unit_key(self) -> str:
        """Primera unidad de la ruta que aún no está dominada."""
        from .route import ROUTE
        done = {k for k, p in self.all_unit_progress().items() if p["status"] == "mastered"}
        for u in ROUTE:
            if u.key not in done:
                return u.key
        return ROUTE[-1].key

    def save_unit_progress(self, unit_key: str, **fields) -> None:
        now = time.time()
        if self.unit_progress(unit_key) is None:
            self.db.ex("INSERT INTO unit_progress(user_id, unit_key, started_at) VALUES(?,?,?)",
                       (self.user_id, unit_key, fields.pop("started_at", now)))
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            vals = tuple(json.dumps(v) if isinstance(v, (list, dict)) else v for v in fields.values())
            self.db.ex(f"UPDATE unit_progress SET {cols} WHERE user_id=? AND unit_key=?", (*vals, self.user_id, unit_key))

    def unit_words_used(self, terms: list[str] | tuple[str, ...]) -> int:
        """Palabras de la unidad que el alumno ya ha usado (provocadas o espontáneas)."""
        wanted = [t.lower() for t in terms]
        if not wanted:
            return 0
        marks = ",".join("?" * len(wanted))
        row = self.db.one(f"SELECT COUNT(*) AS n FROM vocab WHERE user_id=? AND term IN ({marks}) "
                          "AND prompted_uses + spontaneous_uses > 0", (self.user_id, *wanted))
        return int(row["n"]) if row else 0

    def unit_words_state(self, terms: list[str] | tuple[str, ...]) -> dict[str, str]:
        """término → 'unseen' | 'introduced' (presentada, sin usar) | 'used'."""
        out = {}
        for t in terms:
            item = self.vocab_item(t)
            if item is None:
                out[t.lower()] = "unseen"
            else:
                out[t.lower()] = "used" if item["prompted_uses"] + item["spontaneous_uses"] > 0 else "introduced"
        return out

    # --- resumen para prompts y UI --------------------------------------
    def snapshot(self) -> dict:
        level, score = self.level()
        prof = self.profile()
        skills = self.skills()
        return {
            "level": level,
            "level_score": score,
            "lessons_completed": self.db.count_completed_sessions(self.user_id),
            "strengths": prof.get("strengths", []),
            "weaknesses": prof.get("weaknesses", []),
            "skills": {k: round(v["mastery"], 2) for k, v in sorted(skills.items(), key=lambda kv: kv[1]["mastery"])},
            "recurring_mistakes": [
                {"original": m["original"], "corrected": m["corrected"], "count": m["count"]}
                for m in self.recurring_mistakes()
            ],
            "vocab": self.vocab_overview()["by_status"],
            "patience": self.patience().to_dict(),
        }
