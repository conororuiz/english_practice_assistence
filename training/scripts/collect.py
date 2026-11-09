"""Fase 1 del entrenamiento: genera contextos reales del tutor simulando clases completas.

- El PROFESOR (respuestas habladas) lo hace el modelo maestro (qwen3.5:9b por Ollama): sus respuestas, tras un control
  de calidad, son las etiquetas de la tarea "chat".
- El ALUMNO lo simula qwen3.5:4b con personas variadas (trabajo, ciudad, familia, aficiones) y nivel de errores.
- Las tareas estructuradas (evaluador, coach, datos personales, ayuda, traducción, resumen) se ejecutan con el 4B sólo
  para que la clase avance; se guardan sus ENTRADAS y se reetiquetan después (relabel.py) con el maestro.

Cada llamada al LLM se guarda en data/raw/<run>.jsonl con el plan de la clase.

    ../.venv/bin/python training/scripts/collect.py --classes 30 --run r1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_class import ERRORS, FakeModels, Sink, real_turns  # noqa: E402
from tutor.ai.coach import COACH_SYSTEM  # noqa: E402,F401
from tutor.ai.conversation import visible_text  # noqa: E402
from tutor.ai.facts import FACTS_SYSTEM  # noqa: E402
from tutor.ai.helper import HELP_SYSTEM, TRANSLATE_SYSTEM  # noqa: E402
from tutor.ai.ollama_llm import OllamaLLM, _flatten  # noqa: E402
from tutor.ai.prompts import EVALUATOR_SYSTEM, SUMMARY_SYSTEM  # noqa: E402
from tutor.config import settings  # noqa: E402
from tutor.learning.db import Database  # noqa: E402
from tutor.learning.fluency import turn_metrics  # noqa: E402
from tutor.learning.memory import UserMemory  # noqa: E402
from tutor.learning.planner import plan_roleplay  # noqa: E402
from tutor.learning.roleplays import ROLEPLAYS  # noqa: E402
from tutor.learning.route import ROUTE  # noqa: E402
from tutor.session.lesson import LessonSession  # noqa: E402
import dataclasses  # noqa: E402

OUT = ROOT / "training" / "data" / "raw"

PERSONAS = [
    "a Spanish woman (29) from Valencia who works as a nurse in a public hospital, lives with her boyfriend, has a "
    "dog called Toby, and likes running and series",
    "a Colombian man (41) from Medellín who is an accountant, married with two kids (8 and 12), likes football "
    "and cooking on Sundays",
    "a Mexican woman (35) from Guadalajara who works as a graphic designer from home, single, has a cat, likes "
    "painting and travelling",
    "a Spanish man (52) from Seville who owns a small bakery, has a wife and a son at university, gets up at 4 a.m., "
    "likes fishing",
    "an Argentinian woman (24) from Córdoba who is a university student of engineering and works part-time in a "
    "café, lives with her parents and her sister, plays volleyball",
    "a Spanish man (34) from Madrid who works in IT support, married, one sister who is a nurse, likes cycling "
    "and cooking",
    "a Peruvian man (30) from Lima who works as a software developer for a US company, lives alone, plays video "
    "games and goes to the gym",
    "a Chilean woman (46) from Santiago who is a primary school teacher, divorced, one daughter (15), likes "
    "gardening and reading",
    "a Spanish woman (38) from Bilbao who is a sales manager and travels a lot for work, married, no kids, likes "
    "cooking and the mountains",
    "a Venezuelan man (27) living in Barcelona who works as a waiter and studies marketing online, shares a flat "
    "with two friends, plays the guitar",
    "a Spanish man (61) from Zaragoza who is retired (he was a mechanic), has three grandchildren, walks every "
    "morning and wants to travel to London",
    "a Colombian woman (32) from Bogotá who works as a project manager in a bank, has a husband and a baby, likes "
    "salsa dancing",
]

LEARNER = """You are role-playing {persona}. You are learning English at about B1 level, in a voice class with a \
teacher. Reply ONLY with what you say out loud to the teacher's last message: 1-2 short, simple spoken sentences \
(sometimes only a few words, sometimes 3 sentences), no quotes, no stage directions. Do what the teacher asks (answer \
the question, repeat the sentence, do the exercise). Your answers come from speech recognition, so write them in \
lowercase with little punctuation. {errors} Occasionally hesitate ("umm", "let me think", "eh"). {extra}"""

EXTRAS = [
    "Rarely (1 in 15 turns) say you don't understand.",
    "Sometimes (1 in 8 turns) you don't know a word and say it in Spanish or ask how to say it in English.",
    "Sometimes give very short answers (one to four words).",
    "Sometimes ask the teacher a short question back.",
    "Rarely (1 in 10 turns) say something a bit strange or unclear, like a wrong number or time.",
]


def system_task(system: str) -> str:
    if system.startswith(EVALUATOR_SYSTEM.split("{")[0][:200]):
        return "eval"
    for name, s in (("coach", COACH_SYSTEM), ("facts", FACTS_SYSTEM), ("summary", SUMMARY_SYSTEM),
                    ("help", HELP_SYSTEM), ("translate", TRANSLATE_SYSTEM)):
        if system.startswith(s[:200]):
            return name
    return "other"


STUB = {
    "coach": {"analysis_es": "-", "suggestions": [], "model_answer": "-", "model_answer_why_es": "-",
              "model_answer_how_es": "-"},
    "help": {"starters": ["I think…"], "words": []},
    "translate": {"es": "-"},
}


class RecLLM(OllamaLLM):
    """Profesor = modelo maestro; tareas estructuradas = modelo rápido (sólo para que la clase avance)."""

    def __init__(self, chat_model: str, parse_model: str, url: str, sink: "Recorder") -> None:
        super().__init__(chat_model, url)
        self.parse_model = parse_model
        self.rec = sink

    async def stream_chat(self, system, messages):
        parts: list[str] = []
        t = time.monotonic()
        async for c in super().stream_chat(system, messages):
            parts.append(c)
            yield c
        self.rec.write({"task": "chat", "system": system, "messages": [_flatten(m) for m in messages],
                        "output": "".join(parts), "seconds": round(time.monotonic() - t, 2)})

    async def _parse_once(self, system, user, schema):
        task = system_task(system)
        if task in STUB:  # no influye en el curso de la clase: sólo se guarda la entrada (se etiqueta después)
            self.rec.write({"task": task, "schema": schema.__name__, "system": system, "user": user})
            return schema.model_validate(STUB[task])
        msgs = [{"role": "system", "content": system + "\nAnswer ONLY with JSON that matches the schema."},
                {"role": "user", "content": user}]
        body = self._body(msgs, stream=False, fmt=schema.model_json_schema(), temperature=0.1)
        body["model"] = self.parse_model
        resp = await self._post(body, stream=False)
        data = json.loads(await resp.aread())
        await resp.aclose()
        out = schema.model_validate_json(data["message"]["content"])
        self.rec.write({"task": system_task(system), "schema": schema.__name__, "system": system, "user": user,
                        "draft": data["message"]["content"]})
        return out


class Recorder:
    def __init__(self, path: Path) -> None:
        self.f = path.open("a")
        self.meta: dict = {}

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps({**self.meta, **rec}, ensure_ascii=False) + "\n")
        self.f.flush()


async def learner_says(llm: OllamaLLM, model: str, system: str, history: list[dict]) -> str:
    body = llm._body([{"role": "system", "content": system}] + history[-12:], stream=False, temperature=0.9)
    body["model"] = model
    resp = await llm._post(body, stream=False)
    data = json.loads(await resp.aread())
    await resp.aclose()
    return visible_text(data["message"]["content"]).strip().strip('"') or "sorry can you repeat"


async def run_class(llm: RecLLM, db: Database, learner_model: str, persona: str, errors: str, extra: str,
                    rng: random.Random, roleplay: str | None, replay: list[str] | None) -> None:
    sink = Sink()
    s, plan = settings, None
    if roleplay:
        plan = plan_roleplay(UserMemory(db), roleplay)
        s = dataclasses.replace(settings, lesson_seconds=600)
    session = LessonSession(s, FakeModels(), db, llm, sink, local_llm=True, plan=plan)  # type: ignore[arg-type]
    clock = {"t": 0.0}
    session.elapsed = lambda: clock["t"]  # type: ignore[method-assign]
    session.t0 = time.monotonic()
    plan = session.plan
    llm.rec.meta["plan"] = plan.to_dict()
    print(f"=== clase {plan.lesson_number} · unidad {plan.unit_number}: {plan.title} · {errors} · "
          f"{'roleplay ' + roleplay if roleplay else ''}", flush=True)
    learner_sys = LEARNER.format(persona=persona, errors=ERRORS[errors], extra=extra)
    hist: list[dict] = []
    text, user_turn = "(The learner has joined the voice lesson and is ready.)", None
    turns = 0
    while True:
        session._start_reply(text, user_turn=user_turn)
        r = session.reply
        await r.task
        tutor = visible_text(" ".join(r.spoken))
        clock["t"] += len(tutor.split()) / 2.4 + 0.6
        session.tutor_speaking = False
        turns += 1
        if r.final or clock["t"] > s.lesson_seconds + 25:
            break
        pending = [t for t in session.eval_tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=20)
        if rng.random() < 0.06:
            await session.on_command("help", {})
        elif rng.random() < 0.04:
            await session.on_command("translate", {})
        hist.append({"role": "user", "content": tutor})
        if replay is not None:
            if not replay:
                break
            said = replay.pop(0)
        else:
            said = await learner_says(llm, learner_model, learner_sys, hist)
        hist.append({"role": "assistant", "content": said})
        words = len(said.split())
        speech = words / 1.9
        clock["t"] += 1.5 + speech + 2.0
        metrics = turn_metrics(said, speech, speech + 0.6, 1.5, [0.6] if words > 8 else [], False, [])
        text, user_turn = said, {"text": said, "audio": metrics.to_dict(), "t_start": clock["t"] - speech,
                                 "stuck": False, "completeness": 0.9, "final_silence": 1.4}
    for _ in range(60):  # que terminen ayudas/evaluaciones pendientes
        if not llm._eval_lock.locked():
            break
        await asyncio.sleep(0.5)
    await session.finish("completed")
    print(f"    {turns} turnos del profesor", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", type=int, default=10, help="alumnos simulados (cada uno con su BD)")
    ap.add_argument("--classes-per-learner", type=int, default=3)
    ap.add_argument("--run", default="r1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher", default="qwen3.5:9b")
    ap.add_argument("--fast", default="qwen3.5:4b")
    ap.add_argument("--replay", action="store_true", help="un alumno extra que repite los turnos reales de la clase 1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    rec = Recorder(OUT / f"{a.run}.jsonl")
    llm = RecLLM(a.teacher, a.fast, settings.ollama_url, rec)
    rng = random.Random(a.seed)
    t0 = time.time()
    plans = []
    for i in range(a.learners):
        plans.append(("sim", i))
    if a.replay:
        plans.insert(0, ("replay", -1))
    for kind, i in plans:
        db = Database(Path(tempfile.mkdtemp()) / "sim.db")
        mem = UserMemory(db)
        start = 0 if kind == "replay" else rng.choice([0, 0] + list(range(1, len(ROUTE))))
        for u in ROUTE[:start]:
            mem.save_unit_progress(u.key, status="mastered", lessons=2)
        persona = PERSONAS[(i + a.seed) % len(PERSONAS)]
        errors = rng.choice(["low", "mid", "mid", "high"])
        n = 1 if kind == "replay" else a.classes_per_learner
        for c in range(n):
            roleplay = rng.choice(ROLEPLAYS).key if kind == "sim" and rng.random() < 0.2 else None
            extra = rng.choice(EXTRAS)
            rec.meta = {"run": a.run, "learner": f"{a.run}-{i}", "class": c, "errors": errors}
            try:
                await run_class(llm, db, a.fast, persona, errors, extra, rng, roleplay,
                                real_turns(1, str(ROOT / "data" / "tutor.db")) if kind == "replay" else None)
            except Exception as e:  # noqa: BLE001
                print("    clase fallida:", repr(e), flush=True)
            print(f"    {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
