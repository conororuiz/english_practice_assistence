"""Prueba del frontend real en Chrome (headless) con un micrófono simulado alimentado con un WAV.

Recorre el camino del navegador: stream de micrófono → AudioWorklet (16 kHz) → WebSocket → servidor →
audio del tutor reproducido con Web Audio → playback_done → siguiente turno → resumen en pantalla.

Nota: en macOS, Chrome headless se queda bloqueado en getUserMedia incluso con el dispositivo falso (permisos
de micrófono del sistema), así que `getUserMedia` se sustituye por un MediaStream de Web Audio que reproduce el
WAV. Todo lo demás es el código real de la interfaz.

Requisitos: Google Chrome instalado y el servidor arrancado (idealmente con TUTOR_LESSON_SECONDS=45).
Uso: uv run --with playwright python scripts/browser_e2e.py [--wav mic.wav] [--seconds 70]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANSWERS = ["I'm good, thanks. I had a long day at work.",
           "Yesterday I go to a meeting with my manager and we talk about the budget.",
           "We had a deadline for a new project."]


def make_wav(path: Path) -> None:
    from kokoro_onnx import Kokoro

    from tutor.config import settings

    k = Kokoro(str(settings.models_dir / settings.tts_model), str(settings.models_dir / "voices-v1.0.bin"))
    rng = np.random.default_rng(0)
    parts = [rng.normal(0, 0.002, 48000 * 7)]
    for a in ANSWERS:
        x, sr = k.create(a, voice="am_michael", speed=0.95, lang="en-us")
        parts += [np.interp(np.arange(0, len(x), sr / 48000), np.arange(len(x)), x), rng.normal(0, 0.002, 48000 * 9)]
    sf.write(path, np.concatenate(parts).astype(np.float32), 48000, subtype="PCM_16")


async def main(url: str, wav: Path | None, seconds: int) -> None:
    from playwright.async_api import async_playwright

    if wav is None:
        wav = Path(tempfile.mkdtemp()) / "mic.wav"
        make_wav(wav)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
        page = await browser.new_page()
        await page.route("**/__mic.wav", lambda route: route.fulfill(path=str(wav), content_type="audio/wav"))
        await page.add_init_script("""
          navigator.mediaDevices.getUserMedia = async () => {
            const ctx = new AudioContext();
            const data = await (await fetch('/__mic.wav')).arrayBuffer();
            const buf = await ctx.decodeAudioData(data);
            const src = ctx.createBufferSource(); src.buffer = buf;
            const dest = ctx.createMediaStreamDestination(); src.connect(dest); src.start();
            return dest.stream;
          };""")
        page.on("console", lambda m: print(f"[console.{m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
        ws_frames = {"sent_bin": 0, "recv_bin": 0, "recv_json": []}

        def on_ws(ws):
            ws.on("framesent", lambda f: ws_frames.__setitem__("sent_bin", ws_frames["sent_bin"] + (not isinstance(f, str))))
            ws.on("framereceived", lambda f: ws_frames["recv_json"].append(f) if isinstance(f, str)
                  else ws_frames.__setitem__("recv_bin", ws_frames["recv_bin"] + 1))
        page.on("websocket", on_ws)
        await page.goto(url)
        await page.click("#start")
        for _ in range(seconds):
            await asyncio.sleep(1)
            if await page.locator("#summary").is_visible():
                break
        status = await page.text_content("#status")
        summary = await page.text_content("#summary")
        print(f"\nTramas de micrófono enviadas: {ws_frames['sent_bin']} · tramas de audio del tutor recibidas: {ws_frames['recv_bin']}")
        finals = [f for f in ws_frames["recv_json"] if '"user_final"' in f]
        print(f"Turnos del usuario reconocidos: {len(finals)}")
        for f in finals:
            print("  ", f[:160])
        print(f"Estado: {status!r}\nResumen visible: {bool(summary and summary.strip())}\n{(summary or '')[:400]}")
        await page.screenshot(path=str(Path(tempfile.gettempdir()) / "tutor_summary.png"), full_page=True)
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--seconds", type=int, default=75)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.wav, a.seconds))
