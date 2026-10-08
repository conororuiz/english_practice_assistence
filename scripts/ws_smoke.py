"""Prueba de humo del servidor real: abre el WebSocket como lo haría el navegador, "habla" con audio sintético
en tiempo real y espera el resumen final.  Arranca antes el servidor (idealmente con TUTOR_LESSON_SECONDS=60).

Uso: uv run python scripts/ws_smoke.py [--url ws://127.0.0.1:8765/ws/lesson]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANSWERS = ["I'm good, thanks. A bit tired after work.", "Yesterday I have a long meeting with my team.",
           "We talked about the new project and the deadline.", "I think it is next month, maybe.",
           "No, I have never done something like this before.", "I usually cook dinner and watch a series."]


async def main(url: str) -> None:
    from kokoro_onnx import Kokoro

    from tutor.config import settings

    k = Kokoro(str(settings.models_dir / settings.tts_model), str(settings.models_dir / "voices-v1.0.bin"))
    utts = []
    for a in ANSWERS:
        audio, sr = k.create(a, voice="am_michael", speed=0.95, lang="en-us")
        utts.append(np.interp(np.arange(0, len(audio), sr / 16000), np.arange(len(audio)), audio).astype(np.float32))
    queue: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    t0 = time.monotonic()
    turn_latencies = []
    last_user_end = None

    async with websockets.connect(url, max_size=None, origin=url.replace("ws://", "http://").split("/ws")[0]) as ws:
        async def mic():
            rng = np.random.default_rng(0)
            n = 0
            while not done.is_set():
                try:
                    utt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    utt = None
                frames = [rng.normal(0, 0.002, 512).astype(np.float32)] if utt is None else \
                    [utt[i:i + 512] for i in range(0, len(utt) - 511, 512)]
                for f in frames:
                    await ws.send((np.clip(f, -1, 1) * 32767).astype("<i2").tobytes())
                    await asyncio.sleep(0.032)

        async def play(m):
            await asyncio.sleep(m["audio_seconds"])  # reproducción en tiempo real
            await ws.send(json.dumps({"type": "playback_done", "reply_id": m["reply_id"]}))
            if not m["final"]:
                await asyncio.sleep(0.8)
                await queue.put(utts[play.n % len(utts)])
                play.n += 1
        play.n = 0

        await ws.send(json.dumps({"type": "start"}))
        mic_task = asyncio.create_task(mic())
        first_audio_after_user = None
        async for msg in ws:
            if isinstance(msg, bytes):
                if last_user_end is not None and first_audio_after_user is None:
                    first_audio_after_user = time.monotonic()
                    turn_latencies.append(first_audio_after_user - last_user_end)
                continue
            m = json.loads(msg)
            t = time.monotonic() - t0
            if m["type"] == "hello":
                print(f"[{t:5.1f}] servidor LLM={m['llm']}")
            elif m["type"] == "user_final":
                last_user_end, first_audio_after_user = time.monotonic(), None
                print(f"[{t:5.1f}] 🎤 {m['text']}  (silencio final {m['final_silence']}s, c={m['completeness']})")
            elif m["type"] == "tutor_text":
                print(f"[{t:5.1f}] 🧑‍🏫 {m['text']}")
            elif m["type"] == "reply_end":
                asyncio.create_task(play(m))
            elif m["type"] == "summary":
                print(f"[{t:5.1f}] RESUMEN: {json.dumps(m['summary'], ensure_ascii=False)[:600]}")
                done.set()
                break
        mic_task.cancel()
    if turn_latencies:
        print(f"\nLatencia fin de turno detectado → primer audio del tutor: mediana {np.median(turn_latencies):.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws/lesson")
    asyncio.run(main(ap.parse_args().url))
