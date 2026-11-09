"""Lo que el profesor sabe del alumno: su nombre y datos personales que ha contado en clase (trabajo, ciudad,
familia, aficiones…). Se extraen al terminar cada clase y se usan en las siguientes para que la conversación se
sienta con un profesor que le conoce ("How was your painting last Friday?").

Un modelo pequeño se inventa cosas: cada dato debe compartir palabras con lo que el alumno dijo de verdad. Si no hay
modelo, sólo se guarda el nombre (con una regla).
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from .llm import LLM

log = logging.getLogger(__name__)

MAX_FACTS = 12
MAX_NEW_PER_CLASS = 6

FACTS_SYSTEM = """You read what an English learner said during a class and write down personal facts about them, so \
their teacher remembers them in the next class.

Rules:
- Only facts the learner said EXPLICITLY about themselves or their close people: job, city, family, friends, \
routine, hobbies, plans, likes. Never guess or add anything.
- Each fact: short, in simple English, third person, max 12 words. Example: "Works as a Python developer", \
"Lives in Barranquilla; his best friend lives in Cartagena", "Works only 4 hours on Fridays".
- Ignore the class exercises themselves (example sentences they were asked to change) and anything unclear.
- name: the learner's first name only if they said it, otherwise empty.
- At most 6 facts, the most useful for future conversations first."""

_STOP = {"the", "and", "his", "her", "their", "with", "from", "lives", "works", "likes", "has", "have", "about",
         "learner", "they", "she", "he", "does", "every", "very", "really", "usually", "normally", "often", "also",
         "only", "who", "that", "this", "when", "for", "into", "some", "goes"}


class LearnerFacts(BaseModel):
    name: str = Field(default="", description="The learner's first name, if they said it")
    facts: list[str] = Field(default_factory=list, description="Short personal facts, third person")


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-záéíóúñ0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def grounded(fact: str, learner_text: str) -> bool:
    """El dato debe salir de lo que dijo el alumno: al menos la mitad de sus palabras (y dos como mínimo)."""
    words = _content(fact)
    said = _content(learner_text)
    stems = {w[:5] for w in said}
    hits = [w for w in words if w in said or w[:5] in stems]
    return len(hits) >= min(2, len(words)) and len(hits) * 2 >= len(words)


# La frase sin mayúsculas, pero el nombre con mayúscula ("call me later" no es un nombre).
_NAME = re.compile(r"\b(?i:my name is|my name's|i'm called|call me)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)")
_NOT_NAMES = {"later", "back", "now", "tomorrow", "please", "today", "again", "soon", "tonight"}


def name_from(texts: list[str]) -> str:
    for t in texts:
        m = _NAME.search(t)
        if m and m.group(1).lower() not in _NOT_NAMES:
            return m.group(1).capitalize()
    return ""


def merge_facts(old: list[str], new: list[str]) -> list[str]:
    """Añade los datos nuevos; uno nuevo sustituye al antiguo que dice casi lo mismo. Se quedan los más recientes."""
    out = list(old)
    for f in new:
        fw = _content(f)
        for i, o in enumerate(out):
            ow = _content(o)
            if fw and ow and len(fw & ow) / min(len(fw), len(ow)) >= 0.6:
                out.pop(i)
                break
        out.append(f.strip().rstrip("."))
    return out[-MAX_FACTS:]


class FactExtractor:
    def __init__(self, llm: LLM | None):
        self.llm = llm

    async def extract(self, learner_turns: list[str]) -> LearnerFacts:
        texts = [t for t in learner_turns if t.strip()]
        name = name_from(texts)
        if self.llm is None or not texts:
            return LearnerFacts(name=name)
        said = " ".join(texts)
        try:
            out = await self.llm.parse(FACTS_SYSTEM, json.dumps({"learner_said": texts}, ensure_ascii=False),
                                       LearnerFacts)
        except Exception as e:  # noqa: BLE001 — nunca debe romper el cierre de la clase
            log.warning("Extracción de datos del alumno fallida (%s)", e)
            return LearnerFacts(name=name)
        facts = [f.strip() for f in out.facts if f.strip() and len(f.split()) <= 16 and grounded(f, said)]
        model_name = out.name.strip().split(" ")[0].capitalize() if out.name.strip() else ""
        if model_name and (model_name.lower() in _NOT_NAMES or not re.search(
                rf"\b(?i:name is|name's|i'm|i am|call me|this is)\s+{re.escape(model_name)}\b", said)):
            model_name = ""  # el nombre tiene que haberlo dicho al presentarse ("Later" no es un nombre)
        return LearnerFacts(name=name or model_name, facts=facts[:MAX_NEW_PER_CLASS])
