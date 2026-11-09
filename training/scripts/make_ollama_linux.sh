#!/bin/sh
# PC Linux (Ollama con CUDA, motor llama.cpp): fusiona el adaptador PEFT, convierte a GGUF con llama.cpp
# (~/llama.cpp) y crea el modelo de Ollama cuantizado a Q4_K_M.
# Qwen3.5 en Ollama razona por defecto: llamar con "think": false (ollama_llm.py ya lo hace).
#   sh training/scripts/make_ollama_linux.sh [adaptador] [nombre]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ADAPTER=${1:-$ROOT/training/adapters/tutor-sam-4b}; NAME=${2:-tutor-sam}
OLLAMA=${OLLAMA:-$HOME/.local/ollama/bin/ollama}
cd "$ROOT"
rm -rf training/merged/tutor-sam && mkdir -p training/gguf
training/.venv/bin/python training/scripts/merge_torch.py --base training/base/Qwen3.5-4B --adapter "$ADAPTER" \
  --out training/merged/tutor-sam
PYTHONPATH="$HOME/llama.cpp/gguf-py" training/.venv/bin/python "$HOME/llama.cpp/convert_hf_to_gguf.py" \
  training/merged/tutor-sam --outtype bf16 --outfile training/gguf/tutor-sam-bf16.gguf
# Ollama sólo cuantiza al importar safetensors: con GGUF hay que cuantizar antes (llama-quantize, compilado
# sólo para CPU en ~/llama.cpp/build-q; cmake viene del venv: uv pip install cmake).
Q=$HOME/llama.cpp/build-q/bin/llama-quantize
[ -x "$Q" ] || { (cd "$HOME/llama.cpp" && "$ROOT/training/.venv/bin/cmake" -B build-q -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  && "$ROOT/training/.venv/bin/cmake" --build build-q --target llama-quantize -j "$(nproc)"); }
"$Q" training/gguf/tutor-sam-bf16.gguf training/gguf/tutor-sam-q4.tmp.gguf Q4_K_M
# Ollama usa la plantilla Jinja del GGUF (no el TEMPLATE del Modelfile) y la de Qwen3.5 rechaza notas de sistema a
# mitad de conversación (error 500): se sustituye por la del entrenamiento (fmt.py) → ollama/chat_template.jinja.
PYTHONPATH="$HOME/llama.cpp/gguf-py" training/.venv/bin/python "$HOME/llama.cpp/gguf-py/gguf/scripts/gguf_new_metadata.py" \
  --chat-template "$(cat training/ollama/chat_template.jinja)" --force \
  training/gguf/tutor-sam-q4.tmp.gguf training/gguf/tutor-sam-q4_k_m.gguf && rm training/gguf/tutor-sam-q4.tmp.gguf
cd training/ollama && "$OLLAMA" create "$NAME" -f Modelfile.gguf
"$OLLAMA" list | grep "$NAME"
