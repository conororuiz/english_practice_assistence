"""Tutor simulado (sin red ni clave). Sirve para probar todo el pipeline de voz, la detección de turnos,
las métricas y la persistencia. No es conversacional de verdad: sigue un guion sencillo."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_SCRIPT = {
    "warmup": ["Hi! I'm Sam, your English teacher. Nice to see you. How are you feeling today?"],
    "teach": [
        "Today we're going to learn the present simple for routines. <es>Con he o she, el verbo lleva s: she works.</es> "
        "For example: My brother works in a hospital. Can you tell me what someone in your family does?",
        "Good. A useful word is commute <vocab term=\"to commute\" es=\"ir y volver del trabajo\"/> — <es>ir y volver del "
        "trabajo</es>. I commute by train. How do you commute to work? <elicit term=\"to commute\"/>",
    ],
    "produce": [
        "Great work! Now let's use it in a real conversation. What does your manager do every day?",
        "I see. And what time does your day usually start?",
        "Interesting. What do you usually do on your day off?",
    ],
    "review": ["Before we finish, let's review one thing from today. Can you tell me again what your sister does?"],
    "wrapup": ["Great work today! We practised the present simple and the word commute. Your summary and your "
               "progress are on the screen. See you next time!"],
}
_EXERCISE = re.compile(r'(?:task now, clearly|give the next exercise|again the exercise): "([^"]+)"')
class MockLLM:
    def __init__(self) -> None:
        self.last_assistant_content: list[dict] | str | None = None
        self._idx: dict[str, int] = {}

    async def stream_chat(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        phase = "produce"
        for m in reversed(messages):
            if m["role"] == "system":
                found = re.search(r"phase=(\w+)", m["content"])
                if found:
                    phase = found.group(1)
                    break
        note = messages[-1]["content"] if messages and isinstance(messages[-1].get("content"), str) else ""
        i = self._idx.get(phase, 0)
        if phase in ("warmup", "teach", "review") and i >= len(_SCRIPT[phase]):
            phase, i = "produce", self._idx.get("produce", 0)  # guion agotado: sigue la conversación
        if phase == "drill":
            found = _EXERCISE.findall(note)
            text = ("Good. " + found[-1]) if found else "Well done, the exercises are finished."
        else:
            lines = _SCRIPT[phase]
            self._idx[phase] = i + 1
            text = lines[i % len(lines)]
        if "seem stuck" in note:
            text = "Take your time. You can start with: 'I think that…'. What do you think?"
        self.last_assistant_content = text
        for tok in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.02)
            yield tok

    async def parse(self, system: str, user: str, schema: type[T], effort: str | None = None) -> T:
        raise NotImplementedError("El modo mock usa los evaluadores locales")
