#!/bin/sh
# Fusiona un adaptador LoRA con su modelo base y crea el modelo de Ollama (motor MLX, 4 bits).
#   sh training/scripts/make_ollama.sh training/adapters/tutor-4b training/base/Qwen3.5-4B [tutor-sam]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ADAPTER=$(cd "$1" && pwd); BASE=$(cd "$2" && pwd); NAME=${3:-tutor-sam}
rm -rf "$ROOT/training/merged/tutor-sam"
"$ROOT/training/.venv/bin/python" "$ROOT/training/scripts/merge.py" --base "$BASE" --adapter "$ADAPTER" \
  --out "$ROOT/training/merged/tutor-sam"
cd "$ROOT/training/ollama" && ollama create "$NAME" -q int4 -f Modelfile
