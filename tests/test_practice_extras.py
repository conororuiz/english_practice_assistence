"""Tanda 3: role-plays profesionales, pronunciación, racha e informe con audio."""

import dataclasses
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pytest

from tutor.ai.prompts import phase_instruction, render_plan
from tutor.learning.db import Database
from tutor.learning.memory import UserMemory
from tutor.learning.planner import plan_roleplay
from tutor.learning.pronunciation import candidate_word, misheard_pairs, pron_tip
from tutor.learning.report import build_report, report_items
from tutor.learning.roleplays import ROLEPLAYS, ROLEPLAYS_BY_KEY, goals_reached


def mem():
    db = Database(":memory:")
    db.get_or_create_user()
    return UserMemory(db)


# ---------------------------------------------------------------- role-plays
def test_every_roleplay_is_complete():
    assert len(ROLEPLAYS) >= 6
    for r in ROLEPLAYS:
        assert r.goals and len(r.phrases) >= 4 and len(r.beats) >= 4 and r.scene_es and r.learner_role_es


def test_roleplay_plan_has_no_unit_teaching_or_drills():
    m = mem()
    m.add_phrase("on Fridays", "the Fridays", "", "", 1, now=1.0)
    plan = plan_roleplay(m, "daily_standup")
    assert plan.mode == "roleplay" and plan.unit == {} and plan.drills == [] and plan.new_vocab == []
    assert plan.talk[0].startswith("Morning!") and plan.deck  # el mazo de frases se repasa también aquí
    system = render_plan(plan, m.snapshot())
    assert "ROLE-PLAY" in system and "Maya" in system and "Today's class" not in system
    assert "Maya" in phase_instruction("produce", plan, [])
    assert "<es>" in phase_instruction("warmup", plan, [])


def test_goals_are_checked_on_what_the_learner_said():
    rp = ROLEPLAYS_BY_KEY["daily_standup"]
    said = ["Yesterday I fixed the login bug.", "Today I'm going to write the tests.", "No blockers for now."]
    assert goals_reached(rp, said) == [True, True, True]
    assert goals_reached(rp, ["Good morning."]) == [False, False, False]


# ---------------------------------------------------------------- pronunciación
def test_pronunciation_candidates_and_tips():
    assert candidate_word("Thought,") == "thought" and candidate_word("um") == "" and candidate_word("42") == ""
    assert "lengua entre los dientes" in pron_tip("thought")
    assert "sin «e» delante" in pron_tip("start")
    assert "«t»" in pron_tip("worked") and "«-id»" in pron_tip("wanted")
    assert "«d» no suena" in pron_tip("Wednesday")
    assert pron_tip("xyzzy")  # siempre hay un consejo


def test_misheard_pairs_from_repeating_a_sentence():
    assert misheard_pairs(["thought"], ["taught"]) == [("thought", "taught")]
    assert misheard_pairs(["makes"], ["dinner"]) == []


def test_pron_words_count_sessions():
    m = mem()
    m.add_pron_word("thought", 1, "taught")
    m.add_pron_word("thought", 1)
    m.add_pron_word("thought", 2)
    m.add_pron_word("comfortable", 2)
    top = m.pron_words()
    assert top[0]["word"] == "thought" and top[0]["sessions"] == 2 and top[0]["count"] == 3 and top[0]["heard"] == "taught"


# ---------------------------------------------------------------- racha
def test_streak_counts_consecutive_days():
    m = mem()
    now = time.mktime(dt.datetime(2026, 11, 6, 20, 0).timetuple())
    for days_ago in (0, 1, 2, 5, 6):
        sid = m.db.create_session(1, 1, {})
        m.db.finish_session(sid, {}, {})
        m.db.ex("UPDATE sessions SET started_at=? WHERE id=?", (now - days_ago * 86400, sid))
    st = m.streak(now)
    assert st["current"] == 3 and st["best"] == 3 and st["today"]
    assert m.streak(now + 86400)["current"] == 3  # ayer hubo clase: la racha sigue viva
    assert m.streak(now + 2 * 86400)["current"] == 0


# ---------------------------------------------------------------- informe con audio
def test_report_embeds_audio_for_each_corrected_sentence():
    summary = {"lesson_number": 3, "level": "B1", "fluency": {"speaking_wpm": 90}, "unit": {"title_es": "Presente simple",
               "score": 60}, "useful_expression": "No worries", "useful_expression_es": "No pasa nada",
               "pronunciation": [{"word": "thought", "heard": "taught", "tip": "th"}],
               "feedback": [{"text": "My father make breakfast.", "feedback": {"suggestions": [
                   {"kind": "correction", "you_said": "father make", "better": "father makes",
                    "sentence": "My father makes breakfast.", "why_es": "Con he lleva -s."}]}}]}
    calls = []

    def tts(text):
        calls.append(text)
        return np.zeros(2400, np.float32)

    assert report_items(summary)[0]["sentence"] == "My father makes breakfast."
    html = build_report(summary, time.time(), tts)
    assert html.count("data:audio/wav;base64,") == 3 and "Dijiste" in html and "Con he lleva -s." in html
    assert calls == ["My father makes breakfast.", "thought", "No worries"]


# ---------------------------------------------------------------- un role-play entero (modelos reales, tutor simulado)
@pytest.mark.models
async def test_full_roleplay_session(tmp_path: Path):
    import asyncio

    from tests.test_session_e2e import MODELS_OK, FakeClient
    from tutor.ai.mock import MockLLM
    from tutor.config import settings
    from tutor.session.lesson import LessonSession
    from tutor.session.models import Models

    if not MODELS_OK:
        pytest.skip("modelos no descargados")
    models = Models.load(settings)
    s = dataclasses.replace(settings, lesson_seconds=60, data_dir=tmp_path)
    db = Database(tmp_path / "t.db")
    client = FakeClient(models)
    plan = plan_roleplay(UserMemory(db), "daily_standup")
    session = LessonSession(s, models, db, MockLLM(), client, use_llm_for_analysis=False, plan=plan)
    client.session = session
    await session.start()
    mic = asyncio.create_task(client.mic_loop())
    await asyncio.wait_for(client.done.wait(), timeout=120)
    mic.cancel()
    phases = {m["phase"] for m in client.json if m["type"] == "clock"}
    summary = next(m for m in client.json if m["type"] == "summary")["summary"]
    assert "teach" not in phases and "drill" not in phases and "produce" in phases
    assert summary["mode"] == "roleplay" and len(summary["roleplay"]["goals"]) == 3
    assert summary["streak"]["current"] == 1
    assert db.one("SELECT COUNT(*) AS n FROM unit_progress")["n"] == 0  # no cuenta para la unidad
    lesson = next(m for m in client.json if m["type"] == "lesson")
    assert lesson["plan"]["mode"] == "roleplay" and lesson["plan"]["roleplay"]["key"] == "daily_standup"


def test_roleplay_starts_by_setting_the_scene():
    from tutor.session.lesson import LessonSession

    class S(LessonSession):
        def __init__(self, plan):  # sólo lo que usa _phase
            self.plan, self.s = plan, type("X", (), {"lesson_seconds": 600})()
            self.user_turns, self.phase_floor, self.warmup_replies = [], 0, 0
            self.teach, self.teach_step, self.drills, self.drill_pos = [], 0, [], 0
            self.drill_is_asked = self.drill_retry = False
            self.agent = type("A", (), {"last_phase": None})()

        def elapsed(self):
            return 5.0

    sess = S(plan_roleplay(mem(), "job_interview"))
    assert sess._phase() == "warmup"
    sess.warmup_replies = 1
    assert sess._phase() == "produce"


def test_facts_name_must_come_from_an_introduction():
    import asyncio

    from tutor.ai.facts import FactExtractor, LearnerFacts

    class L:
        async def parse(self, *a, **k):
            return LearnerFacts(name="Later", facts=[])

    out = asyncio.run(FactExtractor(L()).extract(["Maybe my sister call me later."]))
    assert out.name == ""
