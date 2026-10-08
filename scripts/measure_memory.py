"""Mide la RAM del backend: tras cargar los modelos y en el peor caso de uso (turno largo de 30 s transcrito,
Smart Turn, varias frases de TTS en paralelo con STT, y una lección simulada completa).

Uso: uv run python scripts/measure_memory.py
"""

from __future__ import annotations

import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.config import settings  # noqa: E402
from tutor.session.models import Models  # noqa: E402

proc = psutil.Process()
peak = 0.0
stop = False


def sampler():
    global peak
    while not stop:
        peak = max(peak, proc.memory_info().rss / 1e6)
        time.sleep(0.01)


def main() -> None:
    global stop
    threading.Thread(target=sampler, daemon=True).start()
    base = proc.memory_info().rss / 1e6
    t0 = time.time()
    models = Models.load(settings)
    loaded = proc.memory_info().rss / 1e6
    print(f"Python base: {base:.0f} MB · modelos cargados: {loaded:.0f} MB ({time.time() - t0:.1f}s)")

    long_text = ("Last year I worked on a really big project with my team. We had a lot of problems at the beginning "
                 "because the deadline was very tight, but in the end we delivered everything on time and the client "
                 "was very happy with the result. I learned a lot about planning.")
    audio, sr = models.tts._kokoro.create(long_text, voice="am_michael", speed=0.9, lang="en-us")
    x = np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32)
    x = np.concatenate([x, x])[: 30 * 16000]

    stt_t, tts_t = [], []

    def stt_job():
        for _ in range(3):
            t = time.time()
            models.stt.transcribe(x, context="Tell me about a project.")
            models.smart_turn.probability_complete(x)
            stt_t.append(time.time() - t)

    def tts_job():
        for s in ["That sounds like a great experience!", "A useful word here is deadline — <es>fecha límite</es>.",
                  "What was the hardest part of the project, and how did you deal with it?"] * 2:
            t = time.time()
            models.tts.synthesize(s)
            tts_t.append(time.time() - t)

    a, b = threading.Thread(target=stt_job), threading.Thread(target=tts_job)
    a.start(), b.start(), a.join(), b.join()
    stop = True
    time.sleep(0.05)
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    print(f"STT+SmartTurn de 30 s de audio: {np.mean(stt_t):.2f}s de media · TTS por frase: {np.mean(tts_t):.2f}s")
    print(f"RSS pico durante la carga de trabajo: {peak:.0f} MB (ru_maxrss {maxrss:.0f} MB)")
    print(f"Presupuesto: 8192 MB total del sistema · margen para navegador y SO: {8192 - max(peak, maxrss):.0f} MB")


if __name__ == "__main__":
    main()
