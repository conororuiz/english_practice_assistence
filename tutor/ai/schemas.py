"""Esquemas de salida estructurada de los componentes con IA (validados con Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ErrorCategory = Literal[
    "grammar", "verb_tense", "vocabulary", "word_choice", "preposition", "article", "word_order",
    "naturalness", "pronunciation", "comprehension",
]


class DetectedError(BaseModel):
    original: str = Field(description="The learner's exact wrong fragment (short)")
    corrected: str = Field(description="The natural, correct version of that fragment")
    category: ErrorCategory
    skill_key: str = Field(description="Grammar/skill key, e.g. past_simple, present_perfect, prepositions, articles, word_choice")
    severity: Literal["major", "minor"] = Field(
        description="major = hurts understanding, core grammar/tense, meaning-changing word, or a repeated pattern; minor = communication still works")
    explanation_es: str = Field(description="Very short explanation in Spanish (max ~15 words)")


class ErrorList(BaseModel):
    """Esquema reducido para modelos locales pequeños: sólo los errores (el resto lo calcula el código)."""
    errors: list[DetectedError] = Field(default_factory=list)


class VocabUse(BaseModel):
    term: str
    correct: bool


class StructureUse(BaseModel):
    skill_key: str
    correct: bool


class TurnEvaluation(BaseModel):
    errors: list[DetectedError] = Field(default_factory=list)
    target_vocab_used: list[VocabUse] = Field(default_factory=list,
                                              description="Target-vocabulary terms the learner actually used in this turn")
    structures_used: list[StructureUse] = Field(default_factory=list,
                                                description="Grammar structures the learner attempted (correctly or not)")
    good_vocabulary: list[str] = Field(default_factory=list,
                                       description="Notable B1+/B2 words or expressions the learner used well")
    complexity: int = Field(ge=1, le=5, description="1 = single words, 3 = simple sentences with some linking, 5 = complex, well-connected ideas")
    understood_question: bool = Field(description="False if the answer shows the learner misunderstood the tutor")
    likely_asr_issue: bool = Field(default=False, description="True if the transcript looks garbled by speech recognition rather than a learner error")


class Suggestion(BaseModel):
    kind: Literal["correction", "more_natural", "richer_answer"] = Field(
        description="correction = fixes one of the verified errors; more_natural = correct but a native speaker would "
                    "say it differently; richer_answer = a fuller model answer to the teacher's question")
    you_said: str = Field(default="", description="The learner's exact fragment (empty for richer_answer)")
    better: str = Field(description="What to say instead, in English")
    why_es: str = Field(description="In Spanish: WHY this is better (the reason or rule), 1-2 short sentences")
    how_es: str = Field(description="In Spanish: HOW to use it — the pattern and when to use it, 1-2 short sentences")
    example: str = Field(description="A NEW English example sentence using the same pattern (different from 'better')")


class TurnFeedback(BaseModel):
    """Análisis de una respuesta del alumno para la pantalla: qué comunicó y cómo decirlo mejor, explicado."""
    analysis_es: str = Field(description="In Spanish, 1-2 sentences: did they answer the question, what they "
                                         "communicated well, what to improve")
    suggestions: list[Suggestion] = Field(default_factory=list, description="Corrections and more natural phrasing")
    model_answer: str = Field(description="A fuller, natural model answer to the teacher's question (2-3 sentences) "
                                          "that keeps the learner's idea and uses today's structure or words")
    model_answer_why_es: str = Field(description="In Spanish: why this answer works better than theirs")
    model_answer_how_es: str = Field(description="In Spanish: how to build answers like this (the pattern to reuse)")


class SummaryVocab(BaseModel):
    term: str
    meaning_es: str


class SessionSummary(BaseModel):
    fluency_line: str = Field(description="One short line on fluency vs. previous sessions (English, may add a few Spanish words)")
    new_vocabulary: list[SummaryVocab]
    improve: list[str] = Field(description="2-3 short points to improve, e.g. 'Past tense: \"I have went\" → \"I have gone\"'")
    well_done: list[str] = Field(description="1-2 concrete things the learner did well")
    useful_expression: str
    useful_expression_es: str
    tomorrow: str = Field(description="One line: what we'll practice next lesson")
    strengths: list[str] = Field(description="Updated overall strengths of the learner (max 4, short)")
    weaknesses: list[str] = Field(description="Updated overall weaknesses of the learner (max 4, short)")
