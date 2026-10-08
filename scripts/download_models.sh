#!/usr/bin/env bash
# Descarga los modelos locales (~860 MB en disco; +464 MB en Apple Silicon). Idempotente: no vuelve a bajar lo que ya existe.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/faster-whisper-small.en
get() { [ -s "$2" ] && { echo "✓ $2"; return; }; echo "↓ $2"; curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"; }

# Silero VAD v5 (MIT) — 2 MB
get https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx models/silero_vad.onnx
# Smart Turn v3.2 CPU (BSD-2, Pipecat) — 8 MB
get https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx models/smart-turn-v3.2-cpu.onnx
# Kokoro-82M ONNX fp32 (Apache-2.0) + voces — 325 MB + 28 MB
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx models/kokoro-v1.0.onnx
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin models/voices-v1.0.bin
# Whisper small.en en formato CTranslate2 (MIT) — 484 MB
for f in config.json model.bin tokenizer.json vocabulary.txt; do
  get "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/$f" "models/faster-whisper-small.en/$f"
done
# Apple Silicon: Whisper large-v3-turbo cuantizado a 4 bits para MLX (MIT) — 464 MB. Es el reconocedor por defecto
# en estos equipos (mucho más preciso con acento español); small.en queda para otros equipos y para scripts/eval_turns.py.
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  mkdir -p models/whisper-large-v3-turbo-q4-mlx
  for f in config.json weights.npz; do
    get "https://huggingface.co/mlx-community/whisper-large-v3-turbo-q4/resolve/main/$f" "models/whisper-large-v3-turbo-q4-mlx/$f"
  done
fi
echo "Modelos listos en ./models"
