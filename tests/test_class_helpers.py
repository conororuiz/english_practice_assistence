"""Ayudas a petición del alumno (tanda 2): «Ayúdame a responder», Repite / Más despacio / Traducir, «Dila tú» con
nota y el mazo de frases con repetición espaciada."""

import asyncio
import dataclasses
from pathlib import Path

import pytest

from tutor.ai.helper import (AnswerHelp, Helper, HelpWord, Translation, clean_help, compare_repeat, corrected_sentence,
                             last_question, phrase_seen, rule_help)
from tutor.learning.db import Database
from tutor.learning.memory import UserMemory
from tutor.learning.planner import plan_next_lesson
from tutor.learning.srs import DAY


# ---------------------------------------------------------------- ayudar a responder
def test_last_question_is_what_the_learner_has_to_answer():
    assert last_question("Nice! Your father travels a lot. What time does he leave home?") == \
        "What time does he leave home?"
    assert last_question("Tell me about your job.") == "Tell me about your job."


def test_rule_starters_match_the_question_type():
    assert rule_help("Do you work from home?").starters[:2] == ["Yes, I do. For example, …", "No, I don't, because …"]
    assert rule_help("Are you tired?").starters[1] == "No, I'm not, because …"
    assert rule_help("Why do you like it?").starters[0] == "Because …"


def test_model_starters_are_cleaned_and_spanish_is_dropped():
    h = clean_help(AnswerHelp(starters=["I usually start my day by...", "Normalmente yo...", '"Every morning I"',
                                        "I usually start my day by …"],
                              words=[HelpWord(en="commute", es="ir al trabajo"), HelpWord(en="trabajo", es="trabajo")]),
                   "What do you do in the morning?")
    assert h.starters == ["I usually start my day by …", "Every morning I …"]
    assert [w.en for w in h.words] == ["commute"]


def test_too_few_good_starters_are_completed_with_rules():
    h = clean_help(AnswerHelp(starters=["Yo trabajo en…"]), "Do you like your job?")
    assert len(h.starters) >= 2 and all(s.endswith("…") for s in h.starters)


class FakeLLM:
    def __init__(self, out):
        self.out = out
        self.calls = 0

    async def parse(self, system, user, schema, effort=None):
        self.calls += 1
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def test_help_falls_back_to_rules_when_the_model_fails():
    h = asyncio.run(Helper(FakeLLM(RuntimeError("x"))).help("Great. Do you drive to work?"))
    assert h.starters[0].startswith("Yes, I do")


def test_translation_must_be_spanish_and_is_cached():
    llm = FakeLLM(Translation(es="¿A qué hora empiezas a trabajar?"))
    helper = Helper(llm)
    assert asyncio.run(helper.translate("What time do you start work?")) == "¿A qué hora empiezas a trabajar?"
    assert asyncio.run(helper.translate("What time do you start work?")) and llm.calls == 1
    assert asyncio.run(Helper(FakeLLM(Translation(es="What time do you start work?"))).translate(
        "What time do you start work?")) == ""


# ---------------------------------------------------------------- «Dila tú»
def test_corrected_sentence_applies_the_fix_to_the_learner_sentence():
    assert corrected_sentence("I live in Barranquilla. My father he make the breakfast.", "he make", "he makes") == \
        "My father he makes the breakfast."
    assert corrected_sentence("something else", "he make", "he makes") == "he makes"


def test_repeat_score_marks_the_words_that_changed():
    r = compare_repeat("My father makes breakfast every day.", "my father make breakfast every day")
    assert r["missing"] == ["makes"] and r["extra"] == ["make"] and 70 <= r["score"] < 90
    assert compare_repeat("He makes breakfast.", "He makes breakfast")["score"] == 100
    assert compare_repeat("He makes breakfast.", "he makes breakfast and lunch and dinner")["score"] < 100


def test_phrase_seen_needs_the_same_words_in_a_row():
    assert phrase_seen("he makes", "Yes, he makes breakfast")
    assert not phrase_seen("he makes", "he always makes it")


# ---------------------------------------------------------------- mazo de frases
def test_deck_spaced_repetition():
    db = Database(":memory:")
    mem = UserMemory(db)
    db.get_or_create_user()
    now = 1_000_000.0
    assert mem.add_phrase("he makes", "he make", "Con he el verbo lleva -s.", "He makes breakfast.", 1, now=now)
    assert not mem.add_phrase("he makes", "he make", "", "", 1, now=now)  # misma clase: no se duplica
    assert mem.due_phrases(now) == [] and len(mem.due_phrases(now + DAY)) == 1  # vuelve mañana
    mem.phrase_result("he makes", True, 92, now=now + DAY)
    p = mem.phrases()[0]
    assert p["box"] == 1 and p["best_score"] == 92 and p["successes"] == 1
    mem.add_phrase("he makes", "he make", "", "", 2, now=now + 2 * DAY)  # el error se repite en otra clase
    assert mem.phrases()[0]["box"] == 0 and mem.phrases()[0]["failures"] == 1


def test_due_phrases_reach_the_next_lesson_plan():
    db = Database(":memory:")
    mem = UserMemory(db)
    db.get_or_create_user()
    mem.add_phrase("on Fridays", "the Fridays", "Con días se usa 'on'.", "On Fridays I work four hours.", 1,
                   now=1.0)
    plan = plan_next_lesson(mem)
    assert plan.deck and plan.deck[0]["better"] == "on Fridays"


# ---------------------------------------------------------------- en una clase (modelos reales, tutor simulado)
@pytest.mark.models
async def test_buttons_during_a_lesson(tmp_path: Path):
    from tutor.config import settings

    if not (settings.models_dir / settings.tts_model).exists():
        pytest.skip("modelos no descargados")
    from tutor.ai.mock import MockLLM
    from tutor.session.lesson import LessonSession
    from tutor.session.models import Models

    models = Models.load(settings)
    sent: list[dict] = []

    class T:
        async def send_json(self, d):
            sent.append(d)

        async def send_bytes(self, d):
            pass

    s = dataclasses.replace(settings, lesson_seconds=600, data_dir=tmp_path)
    session = LessonSession(s, models, Database(tmp_path / "t.db"), MockLLM(), T(), use_llm_for_analysis=False)
    await session.start()
    for _ in range(200):
        await asyncio.sleep(0.05)
        end = next((m for m in sent if m["type"] == "reply_end"), None)
        if end:
            break
    await session.on_playback_done(end["reply_id"])
    assert session._idle()

    await session.on_command("slower", {})
    for _ in range(100):
        await asyncio.sleep(0.05)
        if sum(m["type"] == "reply_end" for m in sent) == 2:
            break
    extra = [m for m in sent if m["type"] == "tutor_text" and m.get("extra")]
    assert extra and extra[0]["text"] == session.agent.last_tutor_text  # repite lo mismo, sin tocar el historial

    await session.on_command("help", {})
    await session.on_command("translate", {})
    await asyncio.sleep(0.1)
    help_msg = next(m for m in sent if m["type"] == "help")
    assert len(help_msg["starters"]) >= 2
    assert next(m for m in sent if m["type"] == "translation")["text"] == ""  # sin modelo no hay traducción

    # «Dila tú»: el profesor dice la frase, el alumno la repite y recibe su nota (sin respuesta del profesor normal)
    import numpy as np

    async def wait_end(n):
        for _ in range(200):
            await asyncio.sleep(0.05)
            ends = [m for m in sent if m["type"] == "reply_end"]
            if len(ends) >= n:
                return ends[-1]
        raise AssertionError("el profesor no terminó de hablar")

    await session.on_playback_done((await wait_end(2))["reply_id"])
    session.memory.add_phrase("he makes", "he make", "", "My father makes breakfast.", session.session_id)
    await session.on_command("repeat_phrase", {"text": "My father makes breakfast.", "key": "he makes"})
    await session.on_playback_done((await wait_end(3))["reply_id"])
    audio, sr = models.tts._kokoro.create("My father make breakfast.", voice="am_michael", speed=0.95, lang="en-us")
    utt = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
    utt = np.concatenate([np.zeros(8000, np.float32), utt])
    for i in range(0, len(utt) - 511, 512):
        await session.on_audio((utt[i:i + 512] * 32767).astype("<i2").tobytes())
        await asyncio.sleep(0.01)
    quiet = (np.random.default_rng(1).normal(0, 0.002, 512) * 32767).astype("<i2").tobytes()
    for _ in range(1500):  # silencio a ~tiempo real hasta que se cierre el turno
        await session.on_audio(quiet)
        await asyncio.sleep(0.016)
        if any(m["type"] == "repeat_result" for m in sent):
            break
    res = next(m for m in sent if m["type"] == "repeat_result")
    assert res["target"] == "My father makes breakfast." and 0 < res["score"] < 100, res
    assert session.repeat_target is None and not session.user_turns[-1:] or \
        session.user_turns[-1]["text"] != res["said"]  # no cuenta como respuesta de la clase
    assert session.memory.phrases()[0]["best_score"] == res["score"]
    await session.stop("test")
