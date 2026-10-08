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


def test_route_stays_on_unit_until_mastered():
    m = mem()
    engine = LearningEngine(m)
    now = NOW
    plans = []
    for _ in range(4):
        plan, metrics = _fake_session(m, engine, errors_pp=3, now=now)
        plans.append(plan)
        now += DAY
    nxt = plan_next_lesson(m, now)
    # sin dominar el presente simple, la ruta no avanza: misma unidad, clase 5, otro contexto cada vez
    assert [p.unit["key"] for p in plans + [nxt]] == ["present_simple"] * 5
    assert [p.unit_lesson for p in plans + [nxt]] == [1, 2, 3, 4, 5]
    assert len({p.scenario_key for p in plans}) >= 2
    assert not metrics["unit"]["mastered"] and metrics["unit"]["missing"]
    assert nxt.target_grammar[0]["key"] == "present_simple"
    # los errores recurrentes siguen llegando al profesor
    assert any(mk["original"].lower() == "i have went" for mk in nxt.recurring_mistakes)
    # clase 1 presenta 4 palabras, clase 2 las otras 4; las presentadas sin usar se piden después
    assert len(plans[0].new_vocab) == 4 and len(plans[1].new_vocab) == 4
    assert {v["term"] for v in plans[0].new_vocab}.isdisjoint({v["term"] for v in plans[1].new_vocab})
    assert not nxt.new_vocab and nxt.review_vocab
    # tras varias clases sin dominarla, se pide explicarla de otra forma
    assert any("different, simpler way" in n for n in nxt.adaptation_notes)


def _unit_session(m, engine, now, good=True):
    """Clase de la unidad actual: 8 ejercicios respondidos (bien o con error de la estructura) y las palabras usadas."""
    plan = plan_next_lesson(m, now)
    sid = m.db.create_session(1, plan.lesson_number, plan.to_dict())
    vocab = LiveVocabLog()
    for v in plan.new_vocab:
        vocab.introduce(v["term"], v["meaning_es"])
    turns = []
    for i, drill in enumerate(plan.drills[:8]):
        text = "My sister works in a hospital and she gets up early"
        errors = [] if good else [{"original": "she work", "corrected": "she works", "category": "grammar",
                                   "skill_key": "present_simple", "severity": "major", "explanation_es": "s"}]
        ev = {"errors": errors, "target_vocab_used": [], "structures_used": [], "good_vocabulary": [], "complexity": 3,
              "understood_question": True, "likely_asr_issue": False}
        turns.append({"turn_id": None, "text": text, "drill": drill, "eval": ev,
                      "audio": turn_metrics(text, 4.0, 5.0, 1.0, [0.5]).to_dict()})
    for i, v in enumerate(plan.new_vocab + plan.review_vocab):
        vocab.use(v["term"], i + 2, "")
    metrics = engine.finalize(sid, plan, turns, vocab, 900, now=now)
    m.db.finish_session(sid, metrics, {})
    return plan, metrics


def test_route_advances_only_when_mastered():
    m = mem()
    engine = LearningEngine(m)
    p1, m1 = _unit_session(m, engine, NOW)
    assert not m1["unit"]["mastered"]  # una sola clase nunca basta
    assert 0 < m1["unit"]["score"] < 100
    p2, m2 = _unit_session(m, engine, NOW + DAY)
    assert p2.unit["key"] == "present_simple" and p2.unit_lesson == 2
    assert m2["unit"]["mastered"] and m2["unit"]["score"] == 100
    p3 = plan_next_lesson(m, NOW + 2 * DAY)
    assert p3.unit["key"] == "frequency" and p3.unit_number == 2 and p3.unit_lesson == 1
    assert p3.just_mastered == p1.unit["title"]
    # una unidad dominada se repasa unos días después
    p4 = plan_next_lesson(m, NOW + 5 * DAY)
    assert p4.review_unit and p4.review_unit["key"] == "present_simple"


def test_route_blocks_on_low_accuracy():
    m = mem()
    engine = LearningEngine(m)
    _unit_session(m, engine, NOW, good=True)
    _, metrics = _unit_session(m, engine, NOW + DAY, good=False)  # la última clase con errores de la estructura
    assert not metrics["unit"]["mastered"]
    assert any("errores" in x for x in metrics["unit"]["missing"])
    nxt = plan_next_lesson(m, NOW + 2 * DAY)
    assert nxt.unit["key"] == "present_simple"
    assert any("Accuracy" in f for f in nxt.focus)
    # los ejercicios que aún no resolvió bien van primero
    assert nxt.drills[0] not in set(m.unit_progress("present_simple")["drills_done"]) or \
        len(m.unit_progress("present_simple")["drills_done"]) == len(nxt.drills)


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


# ---------------------------------------------------------------- clase guiada (ruta, ejercicios, reglas rápidas)
def test_route_content_is_complete():
    from tutor.learning.curriculum import GRAMMAR, SCENARIOS_BY_KEY
    from tutor.learning.route import ROUTE, TALK, attempts_structure

    assert len(ROUTE) == 20 and len({u.key for u in ROUTE}) == 20
    for u in ROUTE:
        assert u.skill in GRAMMAR and len(u.vocab) == 8 and len(u.drills) >= 8 and len(TALK[u.key]) >= 5
        assert all(s in SCENARIOS_BY_KEY for s in u.scenarios)
        assert attempts_structure(u, u.examples[0]), (u.key, u.examples[0])  # el detector reconoce sus ejemplos


def test_drill_asked_tolerates_paraphrase():
    from tutor.learning.route import drill_asked

    drill = "What time do you usually get up on weekdays?"
    assert drill_asked(drill, "Great! Now, what time do you usually get up on weekdays?")
    assert drill_asked(drill, "Okay. So on weekdays, what time do you get up, usually?")
    assert not drill_asked(drill, "How do you commute to work? Tell me in one full sentence.")


def test_quick_rules_catch_present_simple_errors():
    found = lambda t: [(e.original, e.corrected) for e in rule_based_evaluation(t).errors]  # noqa: E731
    assert found("My sister work in a hospital") == [("my sister work", "my sister works")]
    assert found("she always go to the gym") == [("she always go", "she always goes")]
    assert found("he deal with problems") == [("he deal", "he deals")]
    assert found("she study English") == [("she study", "she studies")]
    assert found("my boss no work on Saturdays") == [("my boss no work", "my boss doesn't work")]
    assert found("I go always to the office") == [("i go always", "i always go")]
    assert found("My boss don't work on Saturdays") == [("my boss don't", "my boss doesn't")]
    for ok in ("Does she work here?", "he can go", "she read a book yesterday", "I want him to help", "let it go"):
        assert found(ok) == [], ok


def test_teach_steps_and_unit_prompt():
    from tutor.ai.prompts import teach_steps, tutor_system

    m = mem()
    plan = plan_next_lesson(m, NOW)
    steps = teach_steps(plan)
    assert len(steps) == 3  # regla + 2 bloques de 2 palabras
    assert plan.unit["explain_es"] in steps[0]
    assert all(v["term"] in " ".join(steps[1:]) for v in plan.new_vocab)
    system = tutor_system(plan, m.snapshot())
    assert "learning path unit 1 of 20" in system and "class 1 of this unit" in system


def test_history_is_compacted_in_one_go():
    from tutor.ai.conversation import HISTORY_KEEP_CHARS, compact_history

    msgs = []
    for i in range(60):
        msgs += [{"role": "user", "content": f"answer {i} " + "x" * 120},
                 {"role": "system", "content": "[note] " + "y" * 60},
                 {"role": "assistant", "content": f"reply {i} " + "z" * 150}]
    out = compact_history(msgs)
    assert sum(len(m["content"]) for m in out) <= HISTORY_KEEP_CHARS
    assert out[0]["role"] == "user" and out[-1]["content"].startswith("reply 59")
    assert sum(m["role"] == "system" for m in out) <= 2  # las notas antiguas se quitan
    assert compact_history(out) == out  # por debajo del límite no se toca (el prefijo se mantiene estable)


def test_bad_class_does_not_block_unit_forever():
    m = mem()
    engine = LearningEngine(m)
    _unit_session(m, engine, NOW, good=True)
    _unit_session(m, engine, NOW + DAY, good=False)
    _, m3 = _unit_session(m, engine, NOW + 2 * DAY, good=True)
    assert not m3["unit"]["mastered"]  # todavía pesa el mal día
    _, m4 = _unit_session(m, engine, NOW + 3 * DAY, good=True)
    assert m4["unit"]["mastered"]  # dos buenas clases seguidas lo superan
    assert m4["unit"]["score_before"] < 100 == m4["unit"]["score"]


def test_stage_directions_are_not_spoken():
    from tutor.ai.conversation import SpokenFilter

    assert SpokenFilter([])("How was your morning? (Pause) Great!") == "How was your morning? Great!"
    assert SpokenFilter([])("I commute (by train) every day.") == "I commute (by train) every day."
