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
    r"natural|we say|rule|with he|with she|verb|tense|you can say|you could say|useful (word|expression|phrase)|means|instead of|correct|remember|"
    r"try (saying|to say)|better to say|careful|<fix|<vocab", re.I)
_ES_SPAN = re.compile(r"<es>(.*?)</es>", re.S)
# Elogios que no se pueden comprobar en el momento (la evaluación llega después) o que son falsos si hay errores.
_GRAMMAR_PRAISE = re.compile(r"\b(us(ing|ed|e)|got|said)\b[^.!?]*\b(perfectly|flawlessly|without (any )?mistakes)\b"
                             r"|\bno mistakes\b|\b(all|every) (the )?(ten |\d+ )?(exercises|answers)( \w+)? (right|correct)\b", re.I)
_PRAISE = re.compile(r"^\W*(perfect|exactly( right)?|excellent|fantastic|brilliant|spot on|well done|correct|that's right|"
                     r"that's (perfect|correct|exactly right))\W*$|\b(perfect(ly)?|exactly right|spot on|flawless)\b", re.I)
_QUOTED = re.compile(r"[\"“‘']([^\"”’']{3,})[\"”’']")


def _same_words(a: str, b: str) -> bool:
    w = lambda t: re.findall(r"[a-z]+", t.lower().replace("’", "'").replace("n't", " not"))  # noqa: E731
    return w(a) == w(b)


class SpokenFilter:
    """Pule la salida del LLM antes de la voz (sobre todo para modelos locales pequeños):
    - quita formato markdown (**negritas**, *cursivas*, `código`, viñetas)
    - el español sólo se permite para enseñar: se descartan los fragmentos <es> que traducen preguntas o frases
      sin relación con una corrección o una palabra nueva (salvo traducir la pregunta si el alumno no la entendió),
      y como máximo se deja uno por respuesta."""

    def __init__(self, target_terms: list[str], allow_question: bool = False, teaching: bool = False,
                 has_errors: bool = False, one_question: bool = False):
        self.terms = [t.lower() for t in target_terms]
        self.has_errors = has_errors  # la respuesta del alumno tiene errores seguros: nada de "Perfect!"
        self.one_question = one_question  # conversación: una sola pregunta por turno (en los ejercicios no se aplica)
        self.asked = False
        self.allow_question = allow_question  # el alumno no entendió: vale traducir la pregunta (una vez)
        self.teaching = teaching  # explicando la regla del día: se permiten dos frases en español, algo más largas
        self.max_spans = 2 if teaching else 1
        self.max_words = 35 if teaching else 20
        self.kept_spanish = 0
        self.prev = ""

    def __call__(self, sentence: str) -> str:
        s = re.sub(r"(\*\*|__|\*|`)", "", sentence)
        s = re.sub(r"^\s*([-•]|\d+\.)\s+", "", s)
        s = re.sub(r"\((?:pause|pauses|smiles?|laughs?|wait[a-z ]*|[A-Z][a-z]+(?: [a-z]+){0,3})\)\s*", "", s)  # acotaciones
        # comentarios del modelo para sí mismo: "(assuming that was your name based on the greeting)"
        s = re.sub(r"\s*\((?:assuming|note|i'm|i am|if |based on|as |this is|in character|out of character)[^)]*\)", "", s,
                   flags=re.I)

        def keep(m: re.Match) -> str:
            inner = m.group(1).strip()
            context = (self.prev + " " + s).lower()
            teaching = self.teaching or bool(_TEACHING_CUE.search(context)) or any(t in context for t in self.terms)
            rescue = self.allow_question and inner.endswith("?")
            if self.kept_spanish >= self.max_spans or len(inner.split()) > self.max_words or \
                    (not rescue and (inner.endswith("?") or not teaching)):
                return ""
            self.kept_spanish += 1
            return m.group(0)

        s = _ES_SPAN.sub(keep, s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        while re.search(r"[.!?,]\s+[.,]", s):  # ". ." que dejan las etiquetas quitadas
            s = re.sub(r"([.!?,])\s+[.,]", r"\1", s)
        s = re.sub(r"\s+([.,!?])", r"\1", s)
        english = re.sub(r"<es>.*?</es>", "", s, flags=re.S).strip()
        if english and self._drop(english):
            return ""
        if english.endswith("?") and len(english.split()) >= 6:  # "3 a.m.? Really?" no cuenta como la pregunta
            self.asked = True
        self.prev = s
        return s

    def _drop(self, english: str) -> bool:
        """Frases que no deben sonar: elogios falsos, correcciones que no cambian nada, una segunda pregunta."""
        if self.asked and self.one_question:
            return True  # ya hizo su pregunta: lo que venga detrás (otra pregunta, otra tarea) sobra
        if _GRAMMAR_PRAISE.search(english):
            return True
        if self.has_errors and _PRAISE.search(english) and len(english.split()) <= 8:
            return True
        quoted = _QUOTED.findall(english)
        if len(quoted) >= 2 and _same_words(quoted[0], quoted[1]) and re.search(
                r"instead|could say|can say|should say|better|say", english, re.I):
            return True
        return False


def visible_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def fmt_clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


# Contexto de Ollama: 8192 tokens. El historial se recorta de golpe (no turno a turno) para que el prefijo del
# prompt siga siendo el mismo la mayoría de turnos y el servidor pueda reutilizar su caché.
HISTORY_MAX_CHARS = 16000  # ~4000 tokens
HISTORY_KEEP_CHARS = 7000


def compact_history(messages: list[dict]) -> list[dict]:
    def size(m: dict) -> int:
        c = m["content"]
        return len(c) if isinstance(c, str) else sum(len(b.get("text", "")) for b in c if isinstance(b, dict))

    if sum(size(m) for m in messages) <= HISTORY_MAX_CHARS:
        return messages
    kept: list[dict] = []
    total = 0
    for m in reversed(messages):
        total += size(m)
        if total > HISTORY_KEEP_CHARS:
            break
        kept.append(m)
    kept.reverse()
    while kept and kept[0]["role"] != "user":  # el historial empieza siempre por un turno del alumno
        kept.pop(0)
    # las notas antiguas ya no sirven: se quitan las de sistema salvo las del final
    return [m for i, m in enumerate(kept) if m["role"] != "system" or i >= len(kept) - 6]


class ConversationAgent:
    def __init__(self, llm: LLM, plan: LessonPlan, snapshot: dict, lesson_seconds: int):
        self.llm = llm
        self.plan = plan
        self.system = tutor_system(plan, snapshot)
        self.lesson_seconds = lesson_seconds
        self.messages: list[dict] = []
        self.last_phase: str | None = None
        self.last_tutor_text = ""
        self.progress_hint = ""  # para el cierre: cómo van los ejercicios de hoy
        self.openers: list[str] = []  # cómo empezó cada respuesta: para no repetir siempre el mismo elogio

    def build_request(self, user_text: str, elapsed: float, phase: str, extra_notes: list[str],
                      practice_items: list[str]) -> list[dict]:
        """Mensajes candidatos para la próxima respuesta (no modifica el historial hasta `commit`)."""
        # Sin reloj: el orquestador decide cuándo cambia el paso (con el reloj a la vista, un modelo pequeño tiende a
        # despedirse antes de tiempo).
        note = [f"[Class step: phase={phase}]"]
        if phase != self.last_phase or phase in ("review", "wrapup"):
            note.append(phase_instruction(phase, self.plan, practice_items, self.progress_hint))
        note.extend(extra_notes)
        recent = list(dict.fromkeys(self.openers[-6:]))
        if recent:
            note.append("[Vary how you react: don't start with or reuse these phrases: "
                        + "; ".join(f"\"{o}\"" for o in recent) + "]")
        self.messages = compact_history(self.messages)
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
        first = re.split(r"(?<=[.!?])\s+", self.last_tutor_text, maxsplit=1)[0].strip()
        if first and len(first.split()) <= 8 and not first.endswith("?"):
            self.openers.append(first)
