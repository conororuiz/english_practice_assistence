"""Correcciones precisas y conversación natural (revisión de la clase real del 2026-10-15)."""

import asyncio

from tutor.ai.conversation import ConversationAgent, SpokenFilter
from tutor.ai.evaluator import fix_explanation, merge_evaluations, narrow_error, rule_based_evaluation, word_request
from tutor.ai.facts import FactExtractor, LearnerFacts, grounded, merge_facts, name_from
from tutor.ai.schemas import DetectedError, TurnEvaluation


def err(o, c, ex="Con he/she el verbo lleva -s.", cat="grammar"):
    return DetectedError(original=o, corrected=c, category=cat, skill_key="x", severity="minor", explanation_es=ex)


# ---------------------------------------------------------------- errores acotados a lo que cambia
def test_whole_sentence_error_is_narrowed_to_the_changed_words():
    (e,) = narrow_error(err("My brother work in a hospital and he likes it.", "My brother works in a hospital and he likes it."))
    assert (e.original, e.corrected) == ("My brother work in", "My brother works in")


def test_context_does_not_cross_a_comma_and_each_change_is_separate():
    out = narrow_error(err("every day I wake up, I get a shower, after I get my breakfast, I start to work.",
                           "every day I wake up, I have a shower, after I have my breakfast, I start to work.",
                           ex="Se usa 'have' en lugar de 'get' aquí."))
    assert [(e.original, e.corrected) for e in out] == [("I get a", "I have a"), ("after I get my", "after I have my")]


def test_rewrites_are_not_reported_as_errors():
    assert narrow_error(err("the first thing in the morning that I do is in the job is to check dependent tasks",
                            "the first thing I do in the morning at work is to check my tasks and plan the day")) == []


def test_explanation_only_kept_for_the_change_it_talks_about():
    out = narrow_error(err("The problem is the time because he needs to go 43 kilometers.",
                           "The problem is time because he needs to commute 43 kilometers.",
                           ex="Necesita usar 'commute' en lugar de 'go'."))
    by = {e.corrected: e.explanation_es for e in out}
    assert "commute" in by["needs to commute 43"] and by["problem is time"] == ""
    assert fix_explanation(out[0]).explanation_es  # se rellena con una explicación genérica en español


def test_precise_rule_error_wins_over_whole_sentence_llm_error():
    said = "my father is other person that in the morning he make the breakfast"
    llm = TurnEvaluation(errors=[err(said, "my father is another person who in the morning makes breakfast")],
                         complexity=3, understood_question=True)
    merged = merge_evaluations(llm, rule_based_evaluation(said), said)
    assert ("he make", "he makes") in [(e.original, e.corrected) for e in merged.errors]
    assert all(len(e.original.split()) <= 8 for e in merged.errors)


def test_prompt_text_leaking_into_a_correction_is_dropped():
    said = "But the Fridays I work only for hours."
    bad = err(said, "But on Fridays I work for a few hours.”,  # I/you/we/they work · he/she/it works")
    assert merge_evaluations(TurnEvaluation(errors=[bad], complexity=3, understood_question=True),
                             rule_based_evaluation(said), said).errors == []


def test_english_explanations_are_replaced_by_spanish():
    e = fix_explanation(err("to say about that", "to say", ex="Slightly unnatural phrasing.", cat="naturalness"))
    assert "natural" in e.explanation_es and "Slightly" not in e.explanation_es


# ---------------------------------------------------------------- lo que dice el profesor
def test_false_praise_is_not_spoken():
    f = SpokenFilter([], has_errors=True)
    assert f("You're using the present simple perfectly.") == ""
    assert f("Perfect!") == ""
    assert f("Nice, your father makes breakfast.") != ""
    assert SpokenFilter([])("You got all ten exercises correct – brilliant!") == ""


def test_correction_that_changes_nothing_is_not_spoken():
    f = SpokenFilter([])
    assert f("Instead of saying “my boss doesn’t work on Saturdays,” you could say: “My boss doesn’t work on Saturdays.”") == ""
    assert f("Instead of “he make”, say “he makes”.") != ""


def test_only_one_question_per_conversation_turn():
    f = SpokenFilter([], one_question=True)
    assert f("That's a busy schedule!") and f("What time do you usually start work?")
    assert f("And what time do you finish?") == ""
    g = SpokenFilter([])  # en los ejercicios no se recorta
    assert g("Is that right?") and g("Now make this sentence negative: 'My boss works on Saturdays.'")


def test_leftover_dots_from_removed_tags():
    assert SpokenFilter([])("To get up significa levantarse. . For example: I get up at seven.") == \
        "To get up significa levantarse. For example: I get up at seven."


def test_word_requests_are_detected():
    assert word_request("and then we are so… so… I don't know how to say lejos.") == "lejos"
    assert word_request("How do you say 'cargo' in English?") == "cargo"
    assert word_request("No sé cómo se dice madrugar, but I wake up early") == "madrugar"
    assert word_request("Normally I start at 3 a.m. and I finish at 1 p.m.") == ""


def test_tutor_is_told_not_to_reuse_openers():
    class Dummy:
        last_assistant_content = None
    from tutor.learning.db import Database
    from tutor.learning.memory import UserMemory
    from tutor.learning.planner import plan_next_lesson
    mem = UserMemory(Database(":memory:"))
    agent = ConversationAgent(Dummy(), plan_next_lesson(mem), mem.snapshot(), 900)
    agent.commit([], "That's a really good description, Christian! Now, what do you do on Fridays?", "drill")
    req = agent.build_request("ok", 10, "drill", [], [])
    assert "That's a really good description, Christian!" in req[-1]["content"]


# ---------------------------------------------------------------- memoria personal
def test_learner_name_and_grounded_facts():
    said = ["Hello! My name is Christian.", "I live in Barranquilla and my best friend lives in Cartagena.",
            "I am a Python developer."]
    assert name_from(said) == "Christian"
    assert grounded("Lives in Barranquilla; his best friend lives in Cartagena", " ".join(said))
    assert not grounded("Has two children and a dog", " ".join(said))


def test_facts_from_model_are_checked_against_what_was_said():
    class FakeLLM:
        async def parse(self, system, user, schema, effort=None):
            return LearnerFacts(name="Christian", facts=["Works as a Python developer", "Is married to Laura"])
    out = asyncio.run(FactExtractor(FakeLLM()).extract(["My name is Christian. I am a Python developer."]))
    assert out.name == "Christian" and out.facts == ["Works as a Python developer"]


def test_new_fact_replaces_the_old_one_about_the_same_thing():
    facts = merge_facts(["Works only 4 hours on Fridays", "Lives in Barranquilla"], ["Works 4 hours on Fridays and paints"])
    assert facts == ["Lives in Barranquilla", "Works 4 hours on Fridays and paints"]


def test_correction_must_not_change_who_does_the_action():
    from tutor.ai.evaluator import plausible_error
    said = "I live in a different city from my best friend because I live in Barranquilla"
    assert not plausible_error(err("friend because I live in", "friend because he lives in"), said)


def test_short_reaction_question_does_not_use_up_the_turn_question():
    f = SpokenFilter([], one_question=True)
    assert f("3 a.m.? Really?") and f("Do you mean that you start work at 3 p.m.?")
