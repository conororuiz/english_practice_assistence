"""Formato de prompt del modelo afinado. Es EXACTAMENTE el de la plantilla del Modelfile de Ollama
(training/ollama/Modelfile): cada mensaje (también las notas de sistema a mitad de conversación) como
<|im_start|>rol\\ncontenido<|im_end|>\\n, y la respuesta tras un bloque de pensamiento vacío."""

from __future__ import annotations

GEN_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END = "<|im_end|>"


def render_prompt(messages: list[dict]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}{END}\n" for m in messages) + GEN_PREFIX


def chat_messages(system: str, messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
