"""Tutor simulado (sin red ni clave). Sirve para probar todo el pipeline de voz, la detección de turnos,
las métricas y la persistencia. No es conversacional de verdad: sigue un guion sencillo."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_SCRIPT = {
    "warmup": ["Hi! I'm Sam, nice to talk to you again. How are you feeling today?"],
    "main": [
        "Nice. How was your day at work?",
        "I see. What made it so busy?",
        "A useful word here is deadline <vocab term=\"deadline\" es=\"fecha límite\"/> — <es>fecha límite</es>. When is the deadline for your project? <elicit term=\"deadline\"/>",
        "That sounds intense. Have you ever worked on a project like this before?",
        "Interesting. What did you learn from it?",
        "And how do you usually relax after a long day?",
    ],
    "practice": ["Before we finish, let's practise one thing from today. Can you tell me about a deadline you had recently?"],
    "wrapup": ["Great work today! You spoke a lot and used the word deadline. Your written summary is on the screen. See you tomorrow!"],
}


class MockLLM:
    def __init__(self) -> None:
        self.last_assistant_content: list[dict] | str | None = None
        self._idx: dict[str, int] = {}

    async def stream_chat(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        phase = "main"
        for m in reversed(messages):
            if m["role"] == "system":
                found = re.search(r"phase=(\w+)", m["content"])
                if found:
                    phase = found.group(1)
                    break
        i = self._idx.get(phase, 0)
        if phase in ("warmup", "practice") and i >= len(_SCRIPT[phase]):
            phase, i = "main", self._idx.get("main", 0)  # guion agotado: sigue la conversación principal
        lines = _SCRIPT[phase]
        self._idx[phase] = i + 1
        text = lines[i % len(lines)]
        if "seem stuck" in (messages[-1].get("content") or "") if messages and isinstance(messages[-1].get("content"), str) else False:
            text = "Take your time. You can start with: 'I think that…'. What do you think?"
        self.last_assistant_content = text
        for tok in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.02)
            yield tok

    async def parse(self, system: str, user: str, schema: type[T], effort: str | None = None) -> T:
        raise NotImplementedError("El modo mock usa los evaluadores locales")
