"""Conversational AI: mantiene el historial, construye las notas de sistema por turno y convierte el
streaming del LLM en frases listas para TTS + eventos silenciosos (<vocab>, <elicit>, <fix>)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from ..learning.planner import LessonPlan
from .llm import LLM
from .prompts import phase_instruction, tutor_system

_SILENT_TAG = re.compile(r"<(vocab|elicit|fix)\b([^>]*)/?>", re.S)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)")


@dataclass
class TagEvent:
    kind: str  # vocab | elicit | fix
    attrs: dict[str, str]


@dataclass
class StreamParser:
    """Recibe trozos de texto en streaming. Devuelve frases completas (con <es> intacto) y eventos de etiquetas."""

    buf: str = ""
    raw: str = ""
    events: list[TagEvent] = field(default_factory=list)

    def feed(self, chunk: str) -> list[str]:
        self.raw += chunk
        self.buf += chunk
        self._extract_tags()
        out = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            sentence, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if re.search(r"\w", re.sub(r"<[^>]*>", "", sentence)):
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        self._extract_tags(final=True)
        rest = re.sub(r"<(?!/?es>)[^>]*>?$", "", self.buf).strip()
        self.buf = ""
        return [rest] if re.search(r"\w", re.sub(r"<[^>]*>", "", rest)) else []

    def _extract_tags(self, final: bool = False) -> None:
        def repl(m: re.Match) -> str:
            self.events.append(TagEvent(m.group(1), dict(_ATTR.findall(m.group(2)))))
            return ""
        self.buf = _SILENT_TAG.sub(repl, self.buf)

    def _find_boundary(self) -> int | None:
        # No cortar si hay una etiqueta a medio llegar ("<voc…") o un <es> abierto.
        for m in _SENTENCE_END.finditer(self.buf):
            end = m.end()
            head = self.buf[:end]
            if head.count("<es>") > head.count("</es>"):
                continue
            if head.rfind("<") > head.rfind(">"):
                continue
            return end
        return None


_TEACHING_CUE = re.compile(
    r"natural|we say|you can say|you could say|useful (word|expression|phrase)|means|instead of|correct|remember|"
    r"try (saying|to say)|better to say|careful|<fix|<vocab", re.I)
_ES_SPAN = re.compile(r"<es>(.*?)</es>", re.S)


class SpokenFilter:
    """Pule la salida del LLM antes de la voz (sobre todo para modelos locales pequeños):
    - quita formato markdown (**negritas**, *cursivas*, `código`, viñetas)
    - el español sólo se permite para enseñar: se descartan los fragmentos <es> que traducen preguntas o frases
      sin relación con una corrección o una palabra nueva (salvo traducir la pregunta si el alumno no la entendió),
      y como máximo se deja uno por respuesta."""

    def __init__(self, target_terms: list[str], allow_question: bool = False):
        self.terms = [t.lower() for t in target_terms]
        self.allow_question = allow_question  # el alumno no entendió: vale traducir la pregunta (una vez)
        self.kept_spanish = 0
        self.prev = ""

    def __call__(self, sentence: str) -> str:
        s = re.sub(r"(\*\*|__|\*|`)", "", sentence)
        s = re.sub(r"^\s*([-•]|\d+\.)\s+", "", s)

        def keep(m: re.Match) -> str:
            inner = m.group(1).strip()
            context = (self.prev + " " + s).lower()
            teaching = bool(_TEACHING_CUE.search(context)) or any(t in context for t in self.terms)
            rescue = self.allow_question and inner.endswith("?")
            if self.kept_spanish >= 1 or len(inner.split()) > 20 or (not rescue and (inner.endswith("?") or not teaching)):
                return ""
            self.kept_spanish += 1
            return m.group(0)

        s = _ES_SPAN.sub(keep, s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        self.prev = s
        return s


def visible_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def fmt_clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class ConversationAgent:
    def __init__(self, llm: LLM, plan: LessonPlan, snapshot: dict, lesson_seconds: int):
        self.llm = llm
        self.plan = plan
        self.system = tutor_system(plan, snapshot)
        self.lesson_seconds = lesson_seconds
        self.messages: list[dict] = []
        self.last_phase: str | None = None
        self.last_tutor_text = ""

    def build_request(self, user_text: str, elapsed: float, phase: str, extra_notes: list[str],
                      practice_items: list[str]) -> list[dict]:
        """Mensajes candidatos para la próxima respuesta (no modifica el historial hasta `commit`)."""
        note = [f"[Lesson clock {fmt_clock(elapsed)} of {fmt_clock(self.lesson_seconds)} · phase={phase}]"]
        if phase != self.last_phase or phase in ("practice", "wrapup"):
            note.append(phase_instruction(phase, self.plan, practice_items))
        note.extend(extra_notes)
        return self.messages + [
            {"role": "user", "content": user_text},
            {"role": "system", "content": "\n".join(note)},
        ]

    async def generate(self, request: list[dict]) -> AsyncIterator[tuple[str, object]]:
        """Produce ("sentence", str) a medida que llegan frases y, al final, ("done", (raw, events))."""
        parser = StreamParser()
        async for chunk in self.llm.stream_chat(self.system, request):
            for s in parser.feed(chunk):
                yield "sentence", s
        for s in parser.flush():
            yield "sentence", s
        yield "done", (parser.raw, parser.events)

    def commit(self, request: list[dict], raw_text: str, phase: str, use_provider_content: bool = True) -> None:
        """Añade al historial la petición usada y la respuesta (bloques completos si el proveedor los da;
        sólo el texto realmente pronunciado si el usuario interrumpió)."""
        content = (self.llm.last_assistant_content if use_provider_content else None) or raw_text or "(…)"
        self.messages = request + [{"role": "assistant", "content": content}]
        self.last_phase = phase
        self.last_tutor_text = visible_text(raw_text)
