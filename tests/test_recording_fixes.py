"""Regresiones de la lección grabada del 2026-10-06 (alucinaciones de Whisper, turnos "atascados" falsos,
errores inventados por el evaluador local y el tutor ignorando un "no entiendo")."""

from tutor.ai.conversation import SpokenFilter
from tutor.ai.evaluator import asks_for_help, merge_evaluations, plausible_error
from tutor.ai.schemas import DetectedError, TurnEvaluation
from tutor.audio.stt import clean_transcript, collapse_repetitions
from tutor.audio.turn_detector import text_completeness


def _err(original: str, corrected: str, category: str = "grammar") -> DetectedError:
    return DetectedError(original=original, corrected=corrected, category=category, skill_key="present_simple",
                         severity="major", explanation_es="x")


def test_repetition_loop_is_collapsed():
    text = " ".join(["¡Vamos!"] * 80) + " in this case, eh... with an example."
    assert clean_transcript(text) == "Vamos! in this case, eh... with an example."
    assert collapse_repetitions("I'm not a fan of the " * 4 + "At the moment") == "I'm not a fan of the At the moment"
    assert collapse_repetitions("I, I make some migrations") == "I, I make some migrations"  # 2 veces: es real


def test_trailing_thank_you_hallucination_is_removed():
    assert clean_transcript("Yes, I enjoy it. Thank you.") == "Yes, I enjoy it."
    assert clean_transcript("At the moment, I do... Work Miration of item. Thank you.") == \
        "At the moment, I do... Work Miration of item."
    assert clean_transcript("Thank you.") == ""
    assert clean_transcript("I want to thank you for the help") == "I want to thank you for the help"


def test_thank_you_and_questions_close_the_turn():
    assert text_completeness("Yes, I enjoy it. Thank you.") >= 0.8  # antes: 0.15 por terminar en "you" → STUCK
    assert text_completeness("I work. Can you repeat?") >= 0.85
    assert text_completeness("I don't know") >= 0.8
    assert text_completeness("I went to the") <= 0.15


def test_help_requests_are_detected():
    for t in ("I don't understand the question.", "I work. Can you repeat?", "Sorry, what do you mean?",
              "Could you say that again?", "I don’t understand"):
        assert asks_for_help(t), t
    assert not asks_for_help("I understand the question, I work in IT.")


def test_contraction_expansion_is_not_an_error():
    said = "Now I'm starting the job."
    assert not plausible_error(_err("Now I'm starting the job.", "Now I am starting the job."), said)
    assert plausible_error(_err("Yesterday I go", "Yesterday I went"), "Yesterday I go to the office")


def test_errors_on_unclear_asr_words_are_dropped_and_duplicates_merged():
    said = "At the moment I do some work of miration of item."
    e = _err("work of miration of item", "work of migrating items", "word_choice")
    assert not plausible_error(e, said, unclear=["Miration"])
    dup = _err("I go to work yesterday", "I went to work yesterday")
    ev = merge_evaluations(TurnEvaluation(errors=[dup, dup], complexity=2, understood_question=True),
                           TurnEvaluation(complexity=2, understood_question=True), "I go to work yesterday")
    assert len(ev.errors) == 1


def test_spanish_question_translation_only_when_rescuing():
    normal = SpokenFilter([])
    assert normal("What kind of job do you have? <es>¿Qué tipo de trabajo tienes?</es>") == \
        "What kind of job do you have?"
    rescue = SpokenFilter([], allow_question=True)
    assert "<es>¿Qué tipo de trabajo tienes?</es>" in rescue("What is your job? <es>¿Qué tipo de trabajo tienes?</es>")
    assert "<es>" not in rescue("Do you like it? <es>¿Te gusta?</es>")  # sólo una


def test_real_thanks_inside_the_answer_are_kept():
    assert clean_transcript("Yes, thank you.") == "Yes, thank you."


def test_curly_apostrophe_used_as_full_stop():
    assert clean_transcript("I work’ Can you repeat’") == "I work. Can you repeat."
    assert clean_transcript("It’s my parents’ house") == "It's my parents' house"


def test_correction_must_be_the_same_sentence_fixed():
    assert not plausible_error(_err("Yes, I enjoyed it.", "Yes, I'm starting a new job."), "Yes, I enjoyed it.")
    assert not plausible_error(_err("I work", "I was working"), "I work. Can you repeat?")


def test_stray_quotes_are_removed():
    assert clean_transcript("I work ' can you repeat? 'iwork.") == "I work can you repeat? iwork."
    assert clean_transcript("I don't know, it's fine") == "I don't know, it's fine"
