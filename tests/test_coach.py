"""Análisis de las respuestas (coach): se queda sólo con lo comprobable y explica cada sugerencia."""

import asyncio

from tutor.ai.coach import Coach, clean_feedback, rule_based_feedback, unit_for, worth_coaching
from tutor.ai.schemas import Suggestion, TurnFeedback

SISTER = {"original": "My sister work", "corrected": "My sister works", "category": "word_choice",
          "skill_key": "verb_tense", "severity": "major", "explanation_es": "Con she, el verbo lleva -s."}
PARTY = {"original": "I made a party", "corrected": "I had a party", "category": "grammar",
         "skill_key": "past_simple", "severity": "minor", "explanation_es": "Con party se usa have."}


def fb(suggestions, model="", why="", how=""):
    return TurnFeedback(analysis_es="Has respondido bien a la pregunta, pero falta la -s.",
                        suggestions=suggestions, model_answer=model, model_answer_why_es=why, model_answer_how_es=how)


def sug(kind, you_said, better, why="Es lo que se dice en inglés con el verbo.",
        how="Se usa para hablar de rutinas con el verbo.", example="She works at home."):
    return Suggestion(kind=kind, you_said=you_said, better=better, why_es=why, how_es=how, example=example)


def test_unit_from_the_correction_not_from_the_labels():
    assert unit_for(SISTER).key == "present_simple"  # categoría y destreza mal puestas por el evaluador
    assert unit_for(PARTY) is None  # las dos frases ya están en pasado: no es un error de pasado simple


def test_curated_rule_replaces_the_models_grammar():
    out = clean_feedback(fb([sug("correction", "", "My sister works", how="Usamos have + participio para todo.")]),
                         "My sister work in a bank", [SISTER])
    (s,) = out.suggestions
    assert s.you_said == "My sister work" and "-s" in s.how_es and "participio" not in s.how_es
    assert s.example and s.example != s.better


def test_invented_corrections_are_dropped_and_verified_errors_kept():
    out = clean_feedback(fb([sug("correction", "in a bank", "at a bank")]), "My sister work in a bank", [SISTER])
    assert [s.better for s in out.suggestions] == ["My sister works"]


def test_verified_error_labelled_more_natural_becomes_a_correction():
    out = clean_feedback(fb([sug("more_natural", "I made a party", "I had a party", how="Con fiestas se usa have.")]),
                         "I made a party with my friends", [PARTY])
    (s,) = out.suggestions
    assert s.kind == "correction" and s.how_es == "Con fiestas se usa have."


def test_lone_explanation_is_paired_with_the_lone_error():
    out = clean_feedback(fb([sug("more_natural", "", "I threw a party", how="Para fiestas se usa have o throw.")]),
                         "I made a party with my friends", [PARTY])
    (s,) = out.suggestions
    assert (s.you_said, s.better, s.how_es) == ("I made a party", "I had a party", "Para fiestas se usa have o throw.")


def test_english_explanations_and_unsaid_fragments_are_dropped():
    out = clean_feedback(fb([sug("more_natural", "I take the bus", "I catch the bus"),
                             sug("more_natural", "I go", "I travel", why="It sounds better.", how="Use it always.")]),
                         "I go to work by bus", [])
    assert out.suggestions == []


def test_model_answer_only_when_it_adds_something_and_is_explained():
    long = "I had a party with my friends on Saturday and we danced a lot."
    out = clean_feedback(fb([], long, "Añade cuándo fue y une las ideas.", "Di qué hiciste y añade cuándo con on."),
                         "I made a party", [PARTY])
    assert out.suggestions[-1].kind == "richer_answer" and out.suggestions[-1].better == long
    out = clean_feedback(fb([], long, "It adds details.", "Say when."), "I made a party", [PARTY])
    assert all(s.kind != "richer_answer" for s in out.suggestions)


def test_without_model_the_feedback_comes_from_the_errors():
    out = asyncio.run(Coach(None).feedback("What does she do?", "My sister work in a bank", [SISTER], None))
    assert out == rule_based_feedback([SISTER])
    assert out.suggestions[0].how_es and out.analysis_es
    assert not worth_coaching("Yes") and worth_coaching("Yes, I do")
