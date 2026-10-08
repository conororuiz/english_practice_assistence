"""Text-to-Speech local con Kokoro-82M (ONNX fp32; ~0.25x tiempo real en un M4).

El tutor marca los fragmentos en español con <es>…</es>; cada fragmento se sintetiza con la voz
y el fonemizador del idioma correcto, para que "En este caso usamos *had*…" no suene con acento inglés.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from kokoro_onnx import Kokoro

OUTPUT_SR = 24000
_ES_SPLIT = re.compile(r"<es>(.*?)</es>", re.S)


@dataclass
class Span:
    text: str
    lang: str  # "en" | "es"


def split_languages(text: str) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    for m in _ES_SPLIT.finditer(text):
        if m.start() > pos:
            spans.append(Span(text[pos : m.start()], "en"))
        spans.append(Span(m.group(1), "es"))
        pos = m.end()
    if pos < len(text):
        spans.append(Span(text[pos:], "en"))
    return [s for s in (Span(s.text.strip(), s.lang) for s in spans) if re.search(r"\w", s.text)]


def strip_markup(text: str) -> str:
    """Texto visible/hablado sin etiquetas."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


class TextToSpeech:
    def __init__(self, model_path: str, voices_path: str, voice_en: str, voice_es: str, speed: float = 1.0):
        self._kokoro = Kokoro(model_path, voices_path)
        self.voice_en = voice_en
        self.voice_es = voice_es
        self.speed = speed

    def synthesize(self, text: str) -> np.ndarray:
        """Devuelve float32 mono a 24 kHz. Admite etiquetas <es>; ignora el resto de etiquetas."""
        chunks: list[np.ndarray] = []
        gap = np.zeros(int(0.08 * OUTPUT_SR), dtype=np.float32)
        for span in split_languages(text):
            clean = strip_markup(span.text)
            if not clean:
                continue
            if span.lang == "es":
                audio, _ = self._kokoro.create(clean, voice=self.voice_es, speed=self.speed, lang="es")
            else:
                audio, _ = self._kokoro.create(clean, voice=self.voice_en, speed=self.speed, lang="en-us")
            chunks.extend([audio.astype(np.float32), gap])
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
