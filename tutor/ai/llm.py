"""Proveedor LLM (remoto). Una interfaz mínima para que el resto del sistema no dependa del proveedor:

  - stream_chat(): respuesta del tutor en streaming (para sintetizar voz frase a frase).
  - parse():       salida estructurada validada (evaluación de turnos, resumen de sesión).
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Reintento en servidor con el modelo recomendado si el clasificador de seguridad rechaza la petición.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


class LLM(Protocol):
    def stream_chat(self, system: str, messages: list[dict]) -> AsyncIterator[str]: ...

    async def parse(self, system: str, user: str, schema: type[T], effort: str | None = None) -> T: ...

    @property
    def last_assistant_content(self) -> list[dict] | str | None: ...


class AnthropicLLM:
    def __init__(self, chat_model: str, chat_effort: str, eval_model: str, eval_effort: str):
        self.client = anthropic.AsyncAnthropic(max_retries=2, timeout=anthropic.Timeout(60.0, connect=5.0))
        self.chat_model = chat_model
        self.chat_effort = chat_effort
        self.eval_model = eval_model
        self.eval_effort = eval_effort
        self.last_assistant_content: list[dict] | str | None = None
        self._text_only_history = False

    async def stream_chat(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming de la respuesta del tutor. Tras terminar, `last_assistant_content` contiene los bloques
        completos (pensamiento + texto) para añadirlos al historial tal cual (historial sólo-append)."""
        self.last_assistant_content = None
        msgs = _text_only(messages) if self._text_only_history else messages
        try:
            async for chunk in self._stream(system, msgs):
                yield chunk
        except anthropic.BadRequestError as e:
            if self._text_only_history:
                raise
            # Si el historial con bloques de pensamiento no es aceptado, seguimos con historial de sólo texto.
            log.warning("Historial rechazado (%s); reintentando con historial de sólo texto", e.message)
            self._text_only_history = True
            async for chunk in self._stream(system, _text_only(messages)):
                yield chunk

    async def _stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        async with self.client.beta.messages.stream(
            model=self.chat_model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.chat_effort},
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            log.warning("Respuesta rechazada por el modelo: %s", final.stop_details)
        blocks = []
        for b in final.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "thinking":
                blocks.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
        self.last_assistant_content = blocks if any(b["type"] == "text" for b in blocks) else None

    async def parse(self, system: str, user: str, schema: type[T], effort: str | None = None) -> T:
        response = await self.client.beta.messages.parse(
            model=self.eval_model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort or self.eval_effort},
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RefusalError(f"Sin salida estructurada (stop_reason={response.stop_reason})")
        return response.parsed_output


def _text_only(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        out.append({"role": m["role"], "content": content})
    return out
