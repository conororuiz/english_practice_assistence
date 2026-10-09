"""Descarga los modelos locales (~830 MB en disco; +464 MB en Apple Silicon). Idempotente: no vuelve a bajar lo que
ya existe. Funciona igual en Windows, Linux y macOS:  uv run python scripts/download_models.py"""

import platform
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
HF = "https://huggingface.co"
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

FILES = [
    # Silero VAD v5 (MIT) — 2 MB
    ("https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx", "silero_vad.onnx"),
    # Smart Turn v3.2 CPU (BSD-2, Pipecat) — 8 MB
    (f"{HF}/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-cpu.onnx"),
    # Kokoro-82M ONNX fp32 (Apache-2.0) + voces — 325 MB + 28 MB
    (f"{KOKORO}/kokoro-v1.0.onnx", "kokoro-v1.0.onnx"),
    (f"{KOKORO}/voices-v1.0.bin", "voices-v1.0.bin"),
]
# Whisper small.en en formato CTranslate2 (MIT) — 484 MB
FILES += [(f"{HF}/Systran/faster-whisper-small.en/resolve/main/{f}", f"faster-whisper-small.en/{f}")
          for f in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")]
# Apple Silicon: Whisper large-v3-turbo cuantizado a 4 bits para MLX (MIT) — 464 MB. Es el reconocedor por defecto
# en estos equipos (mucho más preciso con acento español); small.en queda para otros equipos y para scripts/eval_turns.py.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    FILES += [(f"{HF}/mlx-community/whisper-large-v3-turbo-q4/resolve/main/{f}", f"whisper-large-v3-turbo-q4-mlx/{f}")
              for f in ("config.json", "weights.npz")]


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ {dest.relative_to(MODELS.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"↓ {dest.relative_to(MODELS.parent)}")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
    for url, name in FILES:
        get(url, MODELS / name)
    print("Modelos listos en ./models")
