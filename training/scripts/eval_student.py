"""Fase 4: compara modelos (p. ej. qwen3.5:4b de serie frente a tutor-sam) en el conjunto de test, por el MISMO
camino que usa la app (OllamaLLM: prompts completos + salida estructurada para modelos genéricos; prompts
compactos y JSON validado para el afinado).

Métricas:
- eval: precisión / exhaustividad / F1 de los errores frente a la referencia (gold escrito a mano y maestro),
  falsos errores en turnos correctos, JSON válido.
- coach: % de análisis que pasan el control de calidad (cada error explicado, todo en español…), JSON válido.
- chat: puntuación de reglas del profesor (relabel.chat_score), palabras, primer token.
- latencia por llamada (total y tiempo hasta el primer fragmento en el chat).

    .venv/bin/python training/scripts/eval_student.py --models qwen3.5:4b tutor-sam --out training/data/eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relabel import SCHEMAS, chat_score, coach_ok  # noqa: E402
from tutor.ai.evaluator import _overlaps  # noqa: E402
from tutor.ai.ollama_llm import OllamaLLM, OllamaServer  # noqa: E402
from tutor.ai.schemas import ErrorList  # noqa: E402

SFT = ROOT / "training" / "data" / "sft"


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z']+", s.lower()))


def match(p, r) -> bool:
    return _overlaps(p["original"], r["original"]) and (
        norm(p["corrected"]) == norm(r["corrected"]) or _overlaps(p["corrected"], r["corrected"]))


async def run_model(model: str, items: list[dict], limit: dict) -> dict:
    llm = OllamaLLM(model)
    await llm.warmup()
    res: dict[str, list] = {}
    counts: dict[str, int] = {}
    for x in items:
        task, raw = x["task"], x["raw"]
        key = "gold-eval" if (task == "eval" and x["learner"].startswith("gold")) else task
        if counts.get(key, 0) >= limit.get(key, 10**9):
            continue
        counts[key] = counts.get(key, 0) + 1
        t0 = time.monotonic()
        if task == "chat":
            first, out = None, ""
            async for chunk in llm.stream_chat(raw["system"], raw["messages"]):
                first = first or time.monotonic() - t0
                out += chunk
            res.setdefault("chat", []).append({"score": chat_score(out, raw), "words": len(out.split()),
                                               "ttft": first or 0, "s": time.monotonic() - t0, "out": out})
            continue
        try:
            out = await llm._parse_once(raw["system"], raw["user"], SCHEMAS[task])
            ok = True
        except Exception:  # noqa: BLE001
            out, ok = None, False
        dt = time.monotonic() - t0
        r = {"ok": ok, "s": dt}
        if task == "eval":
            ref = json.loads(x["completion"])["errors"]
            pred = [e.model_dump() for e in out.errors] if ok else []
            r.update(tp=sum(any(match(p, q) for q in ref) for p in pred), np=len(pred), nr=len(ref),
                     hit=sum(any(match(p, q) for p in pred) for q in ref), empty_ref=not ref,
                     pred=[f"{p['original']} -> {p['corrected']}" for p in pred],
                     ref=[f"{q['original']} -> {q['corrected']}" for q in ref])
        elif task == "coach" and ok:
            u = json.loads(raw["user"])
            errs = [{"original": e["you_said"], "corrected": e["correct"], "explanation_es": e["reason_es"],
                     "severity": "major", "category": "grammar", "skill_key": "x"} for e in u["verified_errors"]]
            r["good"] = coach_ok(out, u["learner_answered"], errs)
        res.setdefault(key, []).append(r)
    await llm.unload()
    return summarize(res)


def summarize(res: dict) -> dict:
    out = {}
    for key, rs in res.items():
        if key == "chat":
            out[key] = {"n": len(rs), "score": round(st.mean(r["score"] for r in rs), 2),
                        "bad%": round(100 * sum(r["score"] < -1.5 for r in rs) / len(rs)),
                        "words": round(st.mean(r["words"] for r in rs), 1),
                        "ttft_s": round(st.median(r["ttft"] for r in rs), 2),
                        "total_s": round(st.median(r["s"] for r in rs), 2)}
            continue
        o = {"n": len(rs), "json_ok%": round(100 * sum(r["ok"] for r in rs) / len(rs)),
             "median_s": round(st.median(r["s"] for r in rs), 2)}
        if key in ("eval", "gold-eval"):
            tp, np_, nr, hit = (sum(r[k] for r in rs) for k in ("tp", "np", "nr", "hit"))
            p, rc = tp / max(np_, 1), hit / max(nr, 1)
            clean = [r for r in rs if r["empty_ref"]]
            o.update(precision=round(p, 3), recall=round(rc, 3), f1=round(2 * p * rc / max(p + rc, 1e-9), 3),
                     false_alarm_on_correct=round(sum(r["np"] > 0 for r in clean) / max(len(clean), 1), 3))
            if key == "gold-eval":
                o["examples"] = [(r["ref"], r["pred"]) for r in rs if set(r["ref"]) != set(r["pred"])][:12]
        if key == "coach":
            o["good%"] = round(100 * sum(r.get("good", False) for r in rs) / len(rs))
        out[key] = o
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "training" / "data" / "eval.json"))
    ap.add_argument("--limit", type=json.loads, default='{"chat": 80, "eval": 80, "gold-eval": 100, "coach": 40, '
                                                         '"help": 10, "translate": 10, "facts": 5, "summary": 5}')
    a = ap.parse_args()
    items = [json.loads(line) for line in open(SFT / "test.jsonl")]
    server = OllamaServer("http://127.0.0.1:11434")
    await server.ensure_running()
    report = {}
    for m in a.models:
        print("==", m, flush=True)
        report[m] = await run_model(m, items, a.limit)
        print(json.dumps(report[m], ensure_ascii=False, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
