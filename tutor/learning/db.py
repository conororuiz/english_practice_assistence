"""Persistencia (SQLite). Una sola fuente de verdad del progreso del usuario.

Diseñado multiusuario (user_id en todas las tablas) aunque el MVP use un único usuario local.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT 'learner',
    level           TEXT NOT NULL DEFAULT 'B1',
    level_score     REAL NOT NULL DEFAULT 0.0,      -- 0 = B1 recién llegado; 1 = B1+; 2 = B2 ...
    patience_json   TEXT NOT NULL DEFAULT '{}',     -- PatienceProfile aprendido (pausas del usuario)
    profile_json    TEXT NOT NULL DEFAULT '{}',     -- fortalezas/debilidades narrativas, notas del planner
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    lesson_number   INTEGER NOT NULL,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    plan_json       TEXT NOT NULL,                  -- LessonPlan con el que se ejecutó
    metrics_json    TEXT,                           -- SessionMetrics (fluidez, errores, vocabulario)
    summary_json    TEXT,                           -- feedback mostrado al usuario
    status          TEXT NOT NULL DEFAULT 'active'  -- active | completed | aborted
);

CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    idx             INTEGER NOT NULL,
    role            TEXT NOT NULL,                  -- user | tutor
    text            TEXT NOT NULL,
    phase           TEXT NOT NULL,
    t_start         REAL NOT NULL,                  -- segundos desde el inicio de la lección
    t_end           REAL NOT NULL,
    audio_json      TEXT,                           -- métricas acústicas del turno (pausas, wpm, latencia)
    eval_json       TEXT                            -- TurnEvaluation (solo turnos de usuario)
);

CREATE TABLE IF NOT EXISTS errors (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    session_id      INTEGER NOT NULL,
    turn_id         INTEGER,
    category        TEXT NOT NULL,                  -- grammar | vocabulary | naturalness | pronunciation ...
    skill_key       TEXT NOT NULL,                  -- past_simple, present_perfect, prepositions ...
    severity        TEXT NOT NULL,                  -- major | minor
    original        TEXT NOT NULL,
    corrected       TEXT NOT NULL,
    explanation_es  TEXT,
    pattern_key     TEXT NOT NULL,                  -- clave normalizada para detectar errores recurrentes
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_user_pattern ON errors(user_id, pattern_key);

CREATE TABLE IF NOT EXISTS skills (                 -- dominio gramatical por estructura
    user_id         INTEGER NOT NULL,
    skill_key       TEXT NOT NULL,
    mastery         REAL NOT NULL DEFAULT 0.5,      -- 0..1
    errors_total    INTEGER NOT NULL DEFAULT 0,
    correct_total   INTEGER NOT NULL DEFAULT 0,
    last_practiced  REAL,
    next_review     REAL,
    PRIMARY KEY (user_id, skill_key)
);

CREATE TABLE IF NOT EXISTS vocab (                  -- vocabulario y expresiones con repetición espaciada
    user_id         INTEGER NOT NULL,
    term            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'word',   -- word | expression
    meaning_es      TEXT,
    domain          TEXT,                           -- everyday | work
    box             INTEGER NOT NULL DEFAULT 0,     -- caja Leitner (0 = presentada, aún no usada)
    status          TEXT NOT NULL DEFAULT 'new',    -- new | learning | active | mastered
    next_due        REAL,
    introduced_at   REAL NOT NULL,
    exposures       INTEGER NOT NULL DEFAULT 0,     -- veces que el tutor la usó / presentó
    elicitations    INTEGER NOT NULL DEFAULT 0,     -- veces que el tutor intentó provocarla
    prompted_uses   INTEGER NOT NULL DEFAULT 0,     -- el usuario la usó cuando se le provocó
    spontaneous_uses INTEGER NOT NULL DEFAULT 0,    -- el usuario la usó sin que se la pidieran
    failures        INTEGER NOT NULL DEFAULT 0,
    last_used       REAL,
    PRIMARY KEY (user_id, term)
);

CREATE TABLE IF NOT EXISTS unit_progress (         -- ruta de aprendizaje: progreso por unidad
    user_id         INTEGER NOT NULL,
    unit_key        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'current', -- current | mastered
    lessons         INTEGER NOT NULL DEFAULT 0,     -- clases dadas de esta unidad
    correct         REAL NOT NULL DEFAULT 0,        -- usos correctos acumulados de la estructura
    major           REAL NOT NULL DEFAULT 0,
    minor           REAL NOT NULL DEFAULT 0,
    last_correct    REAL NOT NULL DEFAULT 0,        -- lo mismo, sólo en la última clase
    last_major      REAL NOT NULL DEFAULT 0,
    last_minor      REAL NOT NULL DEFAULT 0,
    drills_done     TEXT NOT NULL DEFAULT '[]',     -- ejercicios ya resueltos bien (no se repiten antes que los nuevos)
    score           REAL NOT NULL DEFAULT 0,        -- 0..100 (100 = dominada)
    started_at      REAL NOT NULL,
    last_lesson_at  REAL,
    mastered_at     REAL,
    last_review_at  REAL,                           -- repaso de una unidad ya dominada
    PRIMARY KEY (user_id, unit_key)
);

CREATE TABLE IF NOT EXISTS vocab_events (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    session_id      INTEGER NOT NULL,
    term            TEXT NOT NULL,
    event           TEXT NOT NULL,                  -- introduced | exposed | elicited | used_prompted | used_spontaneous | failed
    created_at      REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # --- utilidades -----------------------------------------------------
    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def ex(self, sql: str, args: tuple = ()) -> int:
        cur = self.conn.execute(sql, args)
        return cur.lastrowid or 0

    # --- usuarios -------------------------------------------------------
    def get_or_create_user(self, user_id: int = 1) -> sqlite3.Row:
        row = self.one("SELECT * FROM users WHERE id=?", (user_id,))
        if row is None:
            self.ex("INSERT INTO users(id, created_at) VALUES(?, ?)", (user_id, time.time()))
            row = self.one("SELECT * FROM users WHERE id=?", (user_id,))
        assert row is not None
        return row

    def update_user(self, user_id: int, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = tuple(json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values())
        self.ex(f"UPDATE users SET {cols} WHERE id=?", (*vals, user_id))

    # --- sesiones -------------------------------------------------------
    def count_completed_sessions(self, user_id: int) -> int:
        row = self.one("SELECT COUNT(*) AS n FROM sessions WHERE user_id=? AND status='completed'", (user_id,))
        return int(row["n"]) if row else 0

    def create_session(self, user_id: int, lesson_number: int, plan: dict) -> int:
        return self.ex(
            "INSERT INTO sessions(user_id, lesson_number, started_at, plan_json) VALUES(?,?,?,?)",
            (user_id, lesson_number, time.time(), json.dumps(plan)),
        )

    def finish_session(self, session_id: int, metrics: dict, summary: dict, status: str = "completed") -> None:
        self.ex(
            "UPDATE sessions SET ended_at=?, metrics_json=?, summary_json=?, status=? WHERE id=?",
            (time.time(), json.dumps(metrics), json.dumps(summary), status, session_id),
        )

    def recent_sessions(self, user_id: int, limit: int = 10) -> list[dict]:
        rows = self.q(
            "SELECT * FROM sessions WHERE user_id=? AND status='completed' ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in ("plan_json", "metrics_json", "summary_json"):
                d[k.removesuffix("_json")] = json.loads(d.pop(k) or "null")
            out.append(d)
        return out

    def add_turn(self, session_id: int, idx: int, role: str, text: str, phase: str, t_start: float,
                 t_end: float, audio: dict | None = None) -> int:
        return self.ex(
            "INSERT INTO turns(session_id, idx, role, text, phase, t_start, t_end, audio_json) VALUES(?,?,?,?,?,?,?,?)",
            (session_id, idx, role, text, phase, t_start, t_end, json.dumps(audio) if audio else None),
        )

    def set_turn_eval(self, turn_id: int, evaluation: dict) -> None:
        self.ex("UPDATE turns SET eval_json=? WHERE id=?", (json.dumps(evaluation), turn_id))

    def session_turns(self, session_id: int) -> list[dict]:
        rows = self.q("SELECT * FROM turns WHERE session_id=? ORDER BY idx", (session_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["audio"] = json.loads(d.pop("audio_json") or "null")
            d["eval"] = json.loads(d.pop("eval_json") or "null")
            out.append(d)
        return out
