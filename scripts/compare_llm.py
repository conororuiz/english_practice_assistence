"""Compara modelos locales de Ollama con los prompts reales del tutor.

Mide: latencia hasta la primera frase (lo que se oye), tiempo total de respuesta, uso de etiquetas (<es>, <vocab>,
<elicit>, <fix>), calidad de las respuestas (se imprimen para leerlas), precisión de la evaluación de errores sobre
frases con errores conocidos, y RAM del proceso de Ollama.

Uso: uv run python scripts/compare_llm.py qwen3.5:4b gemma3:4b
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.ai.conversation import ConversationAgent, SpokenFilter, visible_text  # noqa: E402
from tutor.ai.evaluator import Evaluator  # noqa: E402
from tutor.ai.ollama_llm import OllamaLLM  # noqa: E402
from tutor.learning.db import Database  # noqa: E402
from tutor.learning.memory import UserMemory  # noqa: E402
from tutor.learning.planner import plan_next_lesson  # noqa: E402

LEARNER = [
    (0.5, "warmup", "(The learner has joined the voice lesson and is ready.)"),
    (40, "warmup", "Hi! I'm fine, thanks. A bit tired, today I have a lot of work."),
    (90, "main", "Yesterday I go to the office very early because we have a meeting with the client."),
    (150, "main", "The meeting was about the new project. The client want more features but we don't have time."),
    (220, "main", "Umm, I think we need more people, or maybe, uh, more time for finish it."),
    (300, "main", "I have went to many meetings like this, it's normal in my job."),
    (430, "practice", "Yes, the deadline is next month."),
    (545, "wrapup", "Thank you, see you tomorrow!"),
]
EVAL_CASES = [
    ("Yesterday I go to the office and I have went to a meeting.", {"go", "went"}),
    ("The client want more features, it depends of the budget.", {"want", "depends of"}),
    ("I am agree with you, we made a meeting last week.", {"am agree", "made a meeting"}),
    ("We had a really productive meeting and finished the report on time.", set()),
]


def ollama_rss_mb() -> float:
    out = subprocess.run(["ps", "-axo", "rss,command"], capture_output=True, text=True).stdout
    return sum(int(l.split()[0]) for l in out.splitlines()[1:] if "ollama" in l) / 1024


async def run(model: str) -> None:
    print(f"\n{'=' * 80}\nMODELO {model}\n{'=' * 80}")
    llm = OllamaLLM(model)
    t = time.perf_counter()
    await llm.warmup()
    print(f"carga: {time.perf_counter() - t:.1f}s · RAM Ollama: {ollama_rss_mb():.0f} MB")
    mem = UserMemory(Database(":memory:"))
    plan = plan_next_lesson(mem)
    plan.new_vocab = [{"term": "deadline", "meaning_es": "fecha límite"}, {"term": "workload", "meaning_es": "carga de trabajo"}]
    agent = ConversationAgent(llm, plan, mem.snapshot(), 600)
    firsts, totals, tags = [], [], []
    for elapsed, phase, text in LEARNER:
        items = ["Error: \"Yesterday I go\" → \"Yesterday I went\" (past_simple)", "Vocabulary not yet used: 'deadline'"]
        req = agent.build_request(text, elapsed, phase, [], items)
        t0 = time.perf_counter()
        first, sentences, raw, events = None, [], "", []
        async for kind, val in agent.generate(req):
            if kind == "sentence":
                first = first or time.perf_counter() - t0
                sentences.append(val)
            else:
                raw, events = val
        total = time.perf_counter() - t0
        agent.commit(req, raw, phase)
        firsts.append(first or total)
        totals.append(total)
        tags += [e.kind for e in events] + (["es"] if "<es>" in raw else [])
        flt = SpokenFilter(plan.all_target_terms)
        spoken = " ".join(x for x in (flt(s) for s in sentences) if x)
        print(f"\n🎤 {text}\n🧑‍🏫 [{phase} · 1ª frase {first or total:.2f}s · total {total:.2f}s] {spoken}")
    print(f"\nLatencia 1ª frase: media {sum(firsts) / len(firsts):.2f}s · máx {max(firsts):.2f}s · "
          f"respuesta completa media {sum(totals) / len(totals):.2f}s")
    print(f"Etiquetas usadas: {sorted(set(tags))} ({len(tags)} en total)")

    ev = Evaluator(llm, errors_only=True)  # modo usado con modelos locales
    hits, expected, false_pos = 0, 0, 0
    t0 = time.perf_counter()
    for sentence, must in EVAL_CASES:
        r = await ev.evaluate("Tell me about your week.", sentence, plan, [])
        found = " ".join(e.original.lower() for e in r.errors)
        got = sum(1 for m in must if m in found)
        hits += got
        expected += len(must)
        false_pos += 0 if must else len(r.errors)
        print(f"  eval: {sentence!r} → {[(e.original, e.corrected, e.severity) for e in r.errors]}")
    print(f"Evaluación: {hits}/{expected} errores esperados detectados · falsos positivos en frase correcta: "
          f"{false_pos} · {(time.perf_counter() - t0) / len(EVAL_CASES):.1f}s por turno")
    print(f"RAM Ollama tras la prueba: {ollama_rss_mb():.0f} MB")


async def main() -> None:
    for m in sys.argv[1:] or ["qwen3.5:4b"]:
        await run(m)


if __name__ == "__main__":
    asyncio.run(main())
