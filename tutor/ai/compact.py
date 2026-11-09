"""Prompts compactos para el modelo local afinado para el tutor (training/, «tutor-sam»).

Ese modelo se entrenó con estos prompts (destilación de contexto): las reglas generales del profesor, del
evaluador y del coach ya están en sus pesos, así que sólo recibe lo que cambia en cada clase y en cada turno. Con
modelos genéricos se siguen enviando los prompts completos.

- Tutor: cabecera corta + el plan de la clase (perfil, unidad, palabras…) tal cual.
- Historial: sin las notas de sistema antiguas y sólo los últimos HISTORY_KEEP mensajes; la instrucción del paso
  actual de la clase (que la app sólo manda al empezar el paso) se copia en la nota de este turno.
- Evaluador, coach, ayuda, traducción, datos personales y resumen: una línea que identifica la tarea.

Lo usan la app (OllamaLLM con compact=True) y el entrenamiento (training/scripts/relabel.py), así el modelo ve
en producción exactamente el mismo formato con el que aprendió.
"""

from __future__ import annotations

import re

HISTORY_KEEP = 12  # mensajes alumno/profesor que se conservan (6 intercambios)

TUTOR_HEADER = ("You are {name}, the learner's personal English teacher, giving a one-to-one class by voice to a "
                "Spanish-speaking adult (level {level}). Follow the bracketed class notes.\n\n")

TASKS = {
    "eval": "TASK: evaluate. Find the real errors in the learner's spoken turn (ASR transcript).",
    "coach": "TASK: coach. Write the on-screen feedback in Spanish for the learner's answer.",
    "help": "TASK: help. Give sentence starters and useful words so the learner can answer.",
    "translate": "TASK: translate. Translate the teacher's English into natural Latin American Spanish.",
    "facts": "TASK: facts. Write down the personal facts the learner said about themselves.",
    "summary": "TASK: summary. Write the end-of-lesson feedback card.",
}

_PHASE = re.compile(r"phase=(\w+)")


def task_of(system: str) -> str | None:
    """Qué tarea estructurada es, por el comienzo de su prompt completo (prompts.py, coach.py, helper.py, facts.py)."""
    from .coach import COACH_SYSTEM
    from .facts import FACTS_SYSTEM
    from .helper import HELP_SYSTEM, TRANSLATE_SYSTEM
    from .prompts import EVALUATOR_SYSTEM, SUMMARY_SYSTEM

    for name, full in (("eval", EVALUATOR_SYSTEM), ("coach", COACH_SYSTEM), ("help", HELP_SYSTEM),
                       ("translate", TRANSLATE_SYSTEM), ("facts", FACTS_SYSTEM), ("summary", SUMMARY_SYSTEM)):
        if system.startswith(full[:120]):
            return name
    return None


def compact_parse_system(system: str) -> str:
    task = task_of(system)
    return TASKS[task] if task else system


def compact_tutor_system(system: str) -> str:
    """TUTOR_SYSTEM completo → cabecera corta + plan (desde «# Learner profile»)."""
    i = system.find("# Learner profile")
    if i < 0 or not system.startswith("You are "):
        return system
    name = re.match(r"You are (\w+)", system).group(1)
    level = (re.search(r"at about level (\S+?)\.", system) or re.search(r"Level: (\S+)", system))
    return TUTOR_HEADER.format(name=name, level=level.group(1) if level else "B1") + system[i:]


def _step_instruction(note: str) -> str:
    """La instrucción del paso («STEP 3 — GUIDED PRACTICE…») dentro de una nota, si la tiene."""
    lines = note.split("\n")
    out, inside = [], False
    for line in lines:
        if line.startswith("STEP "):
            inside = True
        elif inside and line.startswith("["):
            break
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def compact_chat(messages: list[dict], keep: int = HISTORY_KEEP) -> list[dict]:
    """Historial compacto. `messages` acaba con el turno del alumno y la nota de sistema de este turno."""
    if not messages or messages[-1]["role"] != "system":
        return messages
    note = messages[-1]["content"]
    phase = (_PHASE.search(note) or [None, None])[1]
    if phase and "STEP " not in note:
        for m in reversed(messages[:-1]):
            if m["role"] == "system" and (_PHASE.search(m["content"]) or [None, None])[1] == phase:
                step = _step_instruction(m["content"])
                if step:
                    first, _, rest = note.partition("\n")
                    note = first + "\n" + step + ("\n" + rest if rest else "")
                    break
    history = [m for m in messages[:-1] if m["role"] != "system"][-keep:]
    while history and history[0]["role"] != "user":
        history.pop(0)
    return history + [{"role": "system", "content": note}]
