"""Memoria, repetición espaciada, ruta adaptativa y métricas (SQLite en memoria, sin modelos)."""

import time

from tutor.ai.conversation import StreamParser, visible_text
from tutor.ai.evaluator import rule_based_evaluation, term_used
from tutor.audio.tts import split_languages
from tutor.learning.db import Database
from tutor.learning.engine import LearningEngine, LiveVocabLog
from tutor.learning.fluency import count_fillers, fluency_index, session_fluency, turn_metrics
from tutor.learning.memory import UserMemory
from tutor.learning.planner import plan_next_lesson
from tutor.learning.srs import DAY, apply_vocab_event, update_mastery

NOW = 1_800_000_000.0


def mem():
    return UserMemory(Database(":memory:"))


# ---------------------------------------------------------------- SRS
def test_vocab_progression_matches_spaced_schedule():
    it = apply_vocab_event({"status": "new"}, "introduced", NOW)  # día 1
    assert it["box"] == 0 and it["next_due"] == NOW + 1 * DAY
    it = apply_vocab_event(it, "used_prompted", NOW + DAY)  # día 2
    assert it["box"] == 1 and it["status"] == "learning"
    it = apply_vocab_event(it, "used_prompted", NOW + 2 * DAY)
    assert it["next_due"] == NOW + 2 * DAY + 3 * DAY  # ~día 5
    it = apply_vocab_event(it, "used_spontaneous", NOW + 5 * DAY)
    assert it["box"] == 4 and it["status"] == "active"
    it = apply_vocab_event(it, "used_spontaneous", NOW + 15 * DAY)
    assert it["status"] == "mastered"  # usada sola dos veces → menos frecuencia


def test_vocab_failure_resets():
    it = {"status": "active", "box": 4}
    it = apply_vocab_event(it, "failed", NOW)
    assert it["box"] == 1 and it["next_due"] == NOW + DAY


def test_mastery_moves_with_evidence():
    assert update_mastery(0.5, correct=0, major=3, minor=0) < 0.35
    assert update_mastery(0.5, correct=5, major=0, minor=0) > 0.7
    assert update_mastery(0.6, 0, 0, 0) == 0.6


# ---------------------------------------------------------------- fluidez
def test_fillers_and_turn_metrics():
    assert count_fillers("Umm, I went to the, uh, office, you know") == 3
    m = turn_metrics("Umm I went to the office", speech_time=2.0, total_time=4.0, onset_latency=1.2, pauses=[1.5])
    assert m.words == 5 and m.fillers == 1
    assert m.articulation_wpm == 150.0 and m.speaking_wpm == 75.0


def test_session_fluency_and_index():
    turns = [{"text": "We had a deadline for a new project and it was busy",
              "audio": turn_metrics("We had a deadline for a new project and it was busy", 4, 5, 1.0, [0.5]).to_dict()}] * 4
    f = session_fluency(turns, 600)
    assert f["turns"] == 4 and f["mean_turn_words"] == 12
    assert 0 < fluency_index(f) <= 100


# ---------------------------------------------------------------- texto del tutor
def test_stream_parser_sentences_and_tags():
    p = StreamParser()
    chunks = ['Nice! A useful word here is "deadline" — <es>fecha', ' límite. Muy útil.</es> <vocab term="dead',
              'line" es="fecha límite"/> When is the deadline? <elicit term="deadline"/>', " Tell me."]
    out = []
    for c in chunks:
        out += p.feed(c)
    out += p.flush()
    assert out[0] == "Nice!"
    assert "<es>fecha límite. Muy útil.</es>" in out[1]  # no corta dentro de <es>
    assert all("<vocab" not in s and "<elicit" not in s for s in out)
    assert [e.kind for e in p.events] == ["vocab", "elicit"]
    assert p.events[0].attrs == {"term": "deadline", "es": "fecha límite"}
    assert visible_text(out[1]).startswith("A useful word")


def test_split_languages():
    spans = split_languages("You can say 'I had'. <es>Usamos had porque ya pasó.</es> Okay?")
    assert [s.lang for s in spans] == ["en", "es", "en"]


# ---------------------------------------------------------------- evaluador local
def test_rule_based_errors():
    ev = rule_based_evaluation("Yesterday I go to the office and I have went to a meeting, it depends of my boss")
    found = {(e.original, e.corrected, e.severity) for e in ev.errors}
    assert ("yesterday i go", "yesterday i went", "major") in found
    assert ("i have went", "i have gone", "major") in found
    assert any(e.corrected == "depends on" and e.severity == "minor" for e in ev.errors)


def test_term_used_inflections():
    assert term_used("to postpone", "we postponed the meeting")
    assert term_used("deadline", "The deadlines are tight")
    assert term_used("to run behind schedule", "we are running behind schedule")
    assert not term_used("deadline", "we had a dead battery")


# ---------------------------------------------------------------- ruta adaptativa
def _fake_session(m: UserMemory, engine: LearningEngine, errors_pp: int, now: float, used_terms=()):
    plan = plan_next_lesson(m, now)
    sid = m.db.create_session(1, plan.lesson_number, plan.to_dict())
    vocab = LiveVocabLog()
    for v in plan.new_vocab:
        vocab.introduce(v["term"], v["meaning_es"])
    for v in plan.review_vocab:
        vocab.elicit(v["term"], 1)
    turns = []
    for i in range(6):
        text = "I have went to the office and we talked about the project with my team"
        if i < len(used_terms):
            text += f" because of the {used_terms[i]}"
        audio = turn_metrics(text, 4.0, 5.5, 1.5, [0.8]).to_dict()
        errors = [{"original": "I have went", "corrected": "I have gone", "category": "verb_tense",
                   "skill_key": "present_perfect", "severity": "major", "explanation_es": "participio"}] * (1 if i < errors_pp else 0)
        ev = {"errors": errors, "target_vocab_used": [], "structures_used": [{"skill_key": "past_simple", "correct": True}],
              "good_vocabulary": [], "complexity": 3, "understood_question": True, "likely_asr_issue": False}
        turns.append({"turn_id": None, "text": text, "audio": audio, "eval": ev})
    for i, term in enumerate(used_terms):
        vocab.use(term, i + 2, "")
    metrics = engine.finalize(sid, plan, turns, vocab, 600, now=now)
    m.db.finish_session(sid, metrics, {})
    return plan, metrics


def test_adaptive_path_targets_recurring_weakness():
    m = mem()
    engine = LearningEngine(m)
    now = NOW
    plans = []
    for _ in range(4):
        plan, _ = _fake_session(m, engine, errors_pp=3, now=now)
        plans.append(plan)
        now += DAY
    nxt = plan_next_lesson(m, now)
    # tras 4 lecciones con errores frecuentes de present perfect, la siguiente lo trabaja en contexto
    assert nxt.target_grammar[0]["key"] == "present_perfect"
    assert any(mk["original"].lower() == "i have went" for mk in nxt.recurring_mistakes)
    # alterna contextos cotidiano / trabajo
    assert {p.domain for p in plans} == {"everyday", "work"}
    # el vocabulario presentado vuelve a repaso al día siguiente
    assert nxt.review_vocab


def test_improvement_reduces_correction_focus():
    m = mem()
    engine = LearningEngine(m)
    m.update_skill("past_simple", correct=0, major=3, minor=0, now=NOW)
    before = m.skills()["past_simple"]["mastery"]
    # varias sesiones usando bien el pasado
    now = NOW + DAY
    for _ in range(3):
        _fake_session(m, engine, errors_pp=0, now=now)
        now += DAY
    after = m.skills()["past_simple"]["mastery"]
    assert after > before
    assert after >= 0.7
    plan = plan_next_lesson(m, now)
    assert "past_simple" not in [g["key"] for g in plan.target_grammar]


def test_vocab_flow_through_sessions():
    m = mem()
    engine = LearningEngine(m)
    plan, _ = _fake_session(m, engine, 0, NOW)
    term = plan.new_vocab[0]["term"]
    assert m.vocab_item(term)["box"] == 0
    plan2 = plan_next_lesson(m, NOW + DAY)
    assert term in [v["term"] for v in plan2.review_vocab]
    _fake_session(m, engine, 0, NOW + DAY, used_terms=[term])
    item = m.vocab_item(term)
    assert item["box"] >= 1 and item["spontaneous_uses"] + item["prompted_uses"] == 1


def test_live_vocab_prompted_vs_spontaneous():
    v = LiveVocabLog()
    v.introduce("deadline", "fecha límite")
    v.elicit("deadline", 3)
    assert v.use("deadline", 4, "") == "used_prompted"
    v.introduce("workload", "carga")
    assert v.use("workload", 9, "we talked about cats") == "used_spontaneous"
    v.elicit("postpone", 5)
    v.close()
    assert ("postpone", "failed", None) in v.events


# ---------------------------------------------------------------- protecciones para modelos pequeños
def test_spoken_filter_spanish_and_markdown():
    from tutor.ai.conversation import SpokenFilter

    f = SpokenFilter(["deadline"])
    # traducción de una pregunta: fuera
    assert f("What kind of work do you do? <es>¿Qué tipo de trabajo haces?</es>") == "What kind of work do you do?"
    # explicación de una corrección: se queda (sólo una por respuesta)
    out = f("A more natural way is 'I went'. <es>Usamos el pasado porque ya ocurrió.</es>")
    assert "<es>Usamos el pasado" in out
    assert "<es>" not in f("You can say 'deadline'. <es>Fecha límite.</es>")
    # markdown fuera
    g = SpokenFilter([])
    assert g("A useful word is **deadline** and *workload*.") == "A useful word is deadline and workload."
    # traducción de una frase sin enseñanza: fuera
    assert "<es>" not in SpokenFilter([])("My manager is busy. <es>Mi jefe está ocupado.</es>")


def test_merge_llm_and_rule_errors():
    from tutor.ai.evaluator import merge_evaluations
    from tutor.ai.schemas import DetectedError, TurnEvaluation

    llm = TurnEvaluation(errors=[DetectedError(original="Yesterday I go to the office", corrected="Yesterday I went to the office",
                                               category="verb_tense", skill_key="past_simple", severity="major",
                                               explanation_es="pasado")], complexity=3, understood_question=True)
    rules = rule_based_evaluation("Yesterday I go to the office and I have went to a meeting")
    merged = merge_evaluations(llm, rules)
    originals = [e.original.lower() for e in merged.errors]
    assert len([o for o in originals if "go" in o and "went" not in o]) == 1  # sin duplicar "yesterday i go"
    assert "i have went" in originals


def test_implausible_llm_errors_are_dropped():
    from tutor.ai.evaluator import plausible_error
    from tutor.ai.schemas import DetectedError

    def err(o, c):
        return DetectedError(original=o, corrected=c, category="grammar", skill_key="x", severity="major", explanation_es="")
    said = "We had a really productive meeting and finished the report on time."
    assert not plausible_error(err("We had a really productive meeting", "I had a really productive meeting"), said)
    assert not plausible_error(err("we have went", "we have gone"), said)  # no lo dijo
    assert not plausible_error(err("on time", "on time"), said)
    assert plausible_error(err("The client want", "The client wants"), "The client want more features")
