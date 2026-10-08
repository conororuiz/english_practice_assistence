"""Carga única de los modelos locales (compartidos por todas las sesiones) y sus ejecutores dedicados.

Cada modelo pesado tiene un ThreadPoolExecutor de 1 hilo: limita la memoria (nunca hay dos inferencias
del mismo modelo a la vez) y evita bloquear el bucle asyncio que procesa el audio en tiempo real.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import psutil

from ..audio.smart_turn import SmartTurn
from ..audio.stt import MLXSpeechToText, SpeechToText, load_stt
from ..audio.tts import TextToSpeech
from ..audio.vad import SileroVAD
from ..config import Settings

log = logging.getLogger(__name__)


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


@dataclass
class Models:
    vad_session: object
    stt: SpeechToText | MLXSpeechToText
    tts: TextToSpeech
    smart_turn: SmartTurn
    stt_pool: ThreadPoolExecutor
    tts_pool: ThreadPoolExecutor

    @classmethod
    def load(cls, s: Settings) -> "Models":
        d = s.models_dir
        missing = [p for p in (d / s.vad_model, d / s.smart_turn_model, d / s.tts_model, d / "voices-v1.0.bin", d / s.stt_model)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Faltan modelos: {missing}. Ejecuta scripts/download_models.sh")
        t0, base = time.time(), rss_mb()
        vad = SileroVAD.load_session(str(d / s.vad_model))
        smart = SmartTurn(str(d / s.smart_turn_model))
        stt = load_stt(d / s.stt_model, threads=s.stt_threads)
        tts = TextToSpeech(str(d / s.tts_model), str(d / "voices-v1.0.bin"), s.tts_voice_en, s.tts_voice_es, s.tts_speed)
        models = cls(vad, stt, tts, smart, ThreadPoolExecutor(1, thread_name_prefix="stt"),
                     ThreadPoolExecutor(1, thread_name_prefix="tts"))
        models.warmup()
        log.info("Modelos cargados en %.1fs · RSS %.0f MB (+%.0f MB)", time.time() - t0, rss_mb(), rss_mb() - base)
        return models

    def warmup(self) -> None:
        """Primera inferencia de cada modelo (evita la latencia de arranque en el primer turno real)."""
        import numpy as np

        silence = np.zeros(16000, dtype=np.float32)
        self.smart_turn.probability_complete(silence)
        self.stt.transcribe(np.random.default_rng(0).normal(0, 0.01, 16000).astype(np.float32))
        self.tts.synthesize("Hi.")
