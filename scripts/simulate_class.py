"""Simula clases completas SÓLO con texto: el profesor real (LessonSession + el LLM local) y un alumno simulado
por el mismo LLM (hispanohablante B1 que comete errores típicos). Sin audio: el reloj de la clase se simula
(tiempo de habla estimado por palabras), así que una clase de 15 minutos tarda unos pocos minutos.

Sirve para revisar el guion de la clase (pasos, ejercicios, correcciones) y la progresión por la ruta.

    uv run python scripts/simulate_class.py                    # una clase, BD temporal
    uv run python scripts/simulate_class.py --classes 3        # tres clases seguidas (progreso en la ruta)
    uv run python scripts/simulate_class.py --errors high      # alumno con muchos errores
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.ai.conversation import visible_text  # noqa: E402
from tutor.ai.ollama_llm import OllamaLLM, OllamaServer  # noqa: E402
from tutor.config import settings  # noqa: E402
from tutor.learning.db import Database  # noqa: E402
from tutor.learning.fluency import turn_metrics  # noqa: E402
from tutor.session.lesson import LessonSession  # noqa: E402

LEARNER = """You are role-playing a Spanish adult (34, works in IT support in Madrid, married, one sister who is a \
nurse, likes cycling and cooking) who is learning English at about B1 level, in a voice class with a teacher. \
Reply ONLY with what you say out loud to the teacher's last message: 1-2 short, simple spoken sentences, no quotes, \
no stage directions. Do what the teacher asks (answer the question, repeat the sentence, do the exercise). \
{errors} Occasionally hesitate ("umm", "let me think"). Rarely (1 in 15 turns) say you don't understand."""

ERRORS = {
    "low": "Make a grammar mistake only sometimes (1 in 5 sentences).",
    "mid": "Make typical Spanish-speaker mistakes in about half of your sentences: forget the -s with he/she "
           "('my sister work'), use the present for the past ('yesterday I go'), 'I am agree', wrong word order "
           "('I go always'). When the teacher corrects you and asks you to repeat, usually say it correctly.",
    "high": "Make typical Spanish-speaker mistakes in most sentences: forget the -s with he/she ('my sister work', "
            "'he don't'), 'yesterday I go', 'I have 34 years'. When the teacher corrects you, sometimes repeat the "
            "same mistake.",
}


class FakeTTS:
    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(2400, dtype=np.float32)


class FakeModels:
    vad_session = None
    stt = None
    smart_turn = None
    tts = FakeTTS()
    stt_pool = ThreadPoolExecutor(1)
    tts_pool = ThreadPoolExecutor(1)


class Sink:
    def __init__(self) -> None:
        self.json: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.json.append(data)

    async def send_bytes(self, data: bytes) -> None:
        pass


async def learner_says(llm: OllamaLLM, system: str, history: list[dict]) -> str:
    out = ""
    async for chunk in llm.stream_chat(system, history[-12:]):
        out += chunk
    return visible_text(out).strip().strip('"') or "Sorry, can you repeat?"


async def run_class(llm: OllamaLLM, db: Database, errors: str, verbose: bool) -> dict:
    sink = Sink()
    s = settings
    session = LessonSession(s, FakeModels(), db, llm, sink, local_llm=True)  # type: ignore[arg-type]
    clock = {"t": 0.0}
    session.elapsed = lambda: clock["t"]  # type: ignore[method-assign]
    session.t0 = time.monotonic()
    plan = session.plan
    print(f"\n=== Clase {plan.lesson_number} · unidad {plan.unit_number}/{plan.unit_count}: {plan.title} "
          f"(clase {plan.unit_lesson} de la unidad, dominio {plan.unit_status.get('score', 0)}%) ===")
    print(f"    palabras nuevas: {[v['term'] for v in plan.new_vocab]} · repaso: {[v['term'] for v in plan.review_vocab]}")
    learner_sys = LEARNER.format(errors=ERRORS[errors])
    learner_hist: list[dict] = []
    text, user_turn = "(The learner has joined the voice lesson and is ready.)", None
    latencies = []
    while True:
        t_req = time.monotonic()
        session._start_reply(text, user_turn=user_turn)
        r = session.reply
        await r.task
        latencies.append(time.monotonic() - t_req)
        tutor = visible_text(" ".join(r.spoken))
        drill = f" [ej {session.drill_pos + 1}/{len(session.drills)}{' repetir' if session.drill_retry else ''}]" \
            if r.phase == "drill" else ""
        print(f"{int(clock['t']) // 60:02d}:{int(clock['t']) % 60:02d} 🧑‍🏫 [{r.phase}]{drill} {tutor}")
        clock["t"] += len(tutor.split()) / 2.4 + 0.6
        session.tutor_speaking = False
        if r.final or clock["t"] > s.lesson_seconds + 25:
            break
        pending = [t for t in session.eval_tasks if not t.done()]
        if pending:  # en una clase real, la evaluación corre mientras el alumno piensa y habla (~5-8 s)
            await asyncio.wait(pending, timeout=8)
        learner_hist.append({"role": "user", "content": tutor})
        said = await learner_says(llm, learner_sys, learner_hist)
        learner_hist.append({"role": "assistant", "content": said})
        words = len(said.split())
        speech = words / 1.9
        onset = 1.5
        clock["t"] += onset + speech + 2.0  # pensar + hablar + fin de turno
        print(f"{int(clock['t']) // 60:02d}:{int(clock['t']) % 60:02d} 🎤 {said}")
        metrics = turn_metrics(said, speech, speech + 0.6, onset, [0.6] if words > 8 else [], False, [])
        text, user_turn = said, {"text": said, "audio": metrics.to_dict(), "t_start": clock["t"] - speech,
                                 "stuck": False, "completeness": 0.9, "final_silence": 1.4}
    await session.finish("completed")
    summ = session.summary or {}
    u = summ.get("unit") or {}
    print(f"--- fin {int(clock['t']) // 60:02d}:{int(clock['t']) % 60:02d} · latencia media del profesor "
          f"{sum(latencies) / len(latencies):.1f}s")
    if u:
        print(f"--- unidad: {u['title']} · dominio {u['score_before']}% → {u['score']}% · dominada={u['mastered']} · "
              f"ejercicios bien {u['drills_ok']}/{u['drills_tried']} · usos correctos {u['correct']:.0f}, errores "
              f"{u['major']:.0f}+{u['minor']:.0f} · palabras usadas {u['words_used']}/{u['words_total']}")
        print(f"--- falta: {u['missing']}")
        print(f"--- próxima: {summ['next_unit']}")
    if verbose:
        for item in summ.get("improve", []):
            print("    mejorar:", item)
    return summ


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=1)
    ap.add_argument("--errors", choices=list(ERRORS), default="mid")
    ap.add_argument("--db", help="BD a usar (por defecto, una temporal nueva)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    server = OllamaServer(settings.ollama_url)
    await server.ensure_running()
    llm = OllamaLLM(settings.ollama_model, settings.ollama_url)
    await llm.warmup()
    db = Database(a.db or Path(tempfile.mkdtemp()) / "sim.db")
    try:
        for _ in range(a.classes):
            await run_class(llm, db, a.errors, a.verbose)
    finally:
        await llm.unload()
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())
