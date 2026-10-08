"""Métricas de fluidez (deterministas, sin IA): se calculan con los tiempos del VAD y la transcripción."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass

from .curriculum import FILLER_WORDS

_SINGLE_FILLERS = {f for f in FILLER_WORDS if " " not in f and f != "like"}
_MULTI_FILLERS = [f for f in FILLER_WORDS if " " in f]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def count_fillers(text: str) -> int:
    toks = tokenize(text)
    n = sum(1 for t in toks if t in _SINGLE_FILLERS)
    low = " " + " ".join(toks) + " "
    n += sum(low.count(f" {f} ") for f in _MULTI_FILLERS)
    # "like" sólo cuenta como muletilla cuando va rodeado de pausa/coma ("it was, like, huge")
    n += len(re.findall(r",\s*like\s*,", text.lower()))
    return n


@dataclass
class TurnAudioMetrics:
    words: int
    speech_time: float  # segundos con voz
    total_time: float  # desde que empieza a hablar hasta que termina (incluye pausas internas)
    onset_latency: float | None  # tiempo desde que el tutor termina hasta que empieza a responder
    pauses: list[float]
    fillers: int
    articulation_wpm: float  # palabras / tiempo de voz
    speaking_wpm: float  # palabras / tiempo total (penaliza pausas)
    stuck: bool
    unclear_words: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def turn_metrics(text: str, speech_time: float, total_time: float, onset_latency: float | None,
                 pauses: list[float], stuck: bool = False, unclear_words: list[str] | None = None) -> TurnAudioMetrics:
    words = len([t for t in tokenize(text) if t not in _SINGLE_FILLERS])
    return TurnAudioMetrics(
        words=words,
        speech_time=round(speech_time, 2),
        total_time=round(total_time, 2),
        onset_latency=None if onset_latency is None else round(onset_latency, 2),
        pauses=[round(p, 2) for p in pauses],
        fillers=count_fillers(text),
        articulation_wpm=round(words / (speech_time / 60), 1) if speech_time > 0.5 else 0.0,
        speaking_wpm=round(words / (total_time / 60), 1) if total_time > 0.5 else 0.0,
        stuck=stuck,
        unclear_words=unclear_words or [],
    )


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio: diversidad léxica estable frente a la longitud del texto."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i : i + window])) / window for i in range(len(tokens) - window + 1)]
    return sum(ratios) / len(ratios)


def session_fluency(user_turns: list[dict], lesson_seconds: float) -> dict:
    """Agrega las métricas de los turnos del usuario. `user_turns` = [{text, audio: TurnAudioMetrics dict}]"""
    audio = [t["audio"] for t in user_turns if t.get("audio")]
    texts = [t["text"] for t in user_turns]
    tokens = [tok for txt in texts for tok in tokenize(txt) if tok not in _SINGLE_FILLERS]
    total_words = len(tokens)
    speech = sum(a["speech_time"] for a in audio)
    total = sum(a["total_time"] for a in audio)
    pauses = [p for a in audio for p in a["pauses"]]
    onsets = [a["onset_latency"] for a in audio if a.get("onset_latency") is not None]
    fillers = sum(a["fillers"] for a in audio)
    lengths = [a["words"] for a in audio if a["words"] > 0]
    word_counts = {}
    for tok in tokens:
        if len(tok) > 3:
            word_counts[tok] = word_counts.get(tok, 0) + 1
    overused = sorted(((w, c) for w, c in word_counts.items() if c >= 5), key=lambda x: -x[1])[:5]
    return {
        "turns": len(user_turns),
        "total_words": total_words,
        "talk_time_s": round(speech, 1),
        "talk_ratio": round(speech / lesson_seconds, 3) if lesson_seconds else 0.0,
        "articulation_wpm": round(total_words / (speech / 60), 1) if speech > 1 else 0.0,
        "speaking_wpm": round(total_words / (total / 60), 1) if total > 1 else 0.0,
        "mean_turn_words": round(statistics.mean(lengths), 1) if lengths else 0.0,
        "max_turn_words": max(lengths) if lengths else 0,
        "median_onset_s": round(statistics.median(onsets), 2) if onsets else None,
        "mean_pause_s": round(statistics.mean(pauses), 2) if pauses else 0.0,
        "long_pauses_per_min": round(sum(1 for p in pauses if p >= 2.0) / (total / 60), 2) if total > 1 else 0.0,
        "fillers_per_100w": round(100 * fillers / total_words, 1) if total_words else 0.0,
        "lexical_diversity": round(mattr(tokens), 3),
        "stuck_events": sum(1 for a in audio if a.get("stuck")),
        "overused_words": [w for w, _ in overused],
    }


def fluency_index(m: dict) -> float:
    """Índice compuesto 0-100 para tendencias (no se enseña como "nota"; sirve para comparar sesiones)."""
    if not m or not m.get("turns"):
        return 0.0
    wpm = min(m.get("speaking_wpm", 0) / 130, 1.0)  # ~130 wpm con pausas ≈ conversación cómoda
    length = min(m.get("mean_turn_words", 0) / 25, 1.0)
    onset = m.get("median_onset_s")
    onset_s = 1.0 if onset is None else max(0.0, 1 - max(onset - 1.0, 0) / 5)
    pauses = max(0.0, 1 - m.get("long_pauses_per_min", 0) / 4)
    fillers = max(0.0, 1 - m.get("fillers_per_100w", 0) / 15)
    diversity = min(m.get("lexical_diversity", 0) / 0.8, 1.0)
    score = 0.25 * wpm + 0.2 * length + 0.15 * onset_s + 0.15 * pauses + 0.1 * fillers + 0.15 * diversity
    return round(100 * score, 1)
