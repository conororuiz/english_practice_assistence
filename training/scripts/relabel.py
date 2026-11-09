"""Fase 2: etiqueta los contextos recogidos (collect.py) con el modelo maestro servido por mlx_lm.server
(qwen3.5 9B, varias peticiones a la vez) y aplica controles de calidad. Salida: data/labeled/<tarea>.jsonl con
{"task", "prompt", "completion"} ya en el formato exacto del modelo afinado (fmt.py).

- eval: errores del maestro → mismos filtros que la app (acotar al fragmento, plausibilidad, gravedad) → JSON compacto.
- coach: se reconstruye su entrada con los errores verificados nuevos (reglas + maestro), se genera y se descarta si
  no explica cada error verificado, si algo no está en español o si la app tendría que retocarlo.
- chat: mejor de N candidatos (el del maestro en la clase + nuevos) según una puntuación de reglas del tutor.
- help / translate / facts / summary: una generación validada.

    ../.venv/bin/python -m mlx_lm.server --model training/base/Qwen3.5-9B-4bit --port 8090 --decode-concurrency 6
    .venv/bin/python training/scripts/relabel.py --raw training/data/raw/r1.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fmt import END, chat_messages, render_prompt  # noqa: E402
from tutor.ai.coach import clean_feedback, is_spanish  # noqa: E402
from tutor.ai.compact import compact_chat, compact_parse_system, compact_tutor_system  # noqa: E402
from tutor.ai.conversation import visible_text  # noqa: E402
from tutor.ai.evaluator import (_LEAK, _overlaps, fix_explanation, merge_evaluations, narrow_error,  # noqa: E402
                                plausible_error, rule_based_evaluation, upgrade_unit_errors)
from tutor.ai.facts import LearnerFacts, grounded  # noqa: E402
from tutor.ai.helper import AnswerHelp, Translation  # noqa: E402
from tutor.ai.ollama_llm import extract_json, parse_messages  # noqa: E402
from tutor.ai.schemas import ErrorList, SessionSummary, TurnEvaluation, TurnFeedback  # noqa: E402
from tutor.learning.planner import LessonPlan  # noqa: E402

SCHEMAS = {"eval": ErrorList, "coach": TurnFeedback, "facts": LearnerFacts, "help": AnswerHelp,
           "translate": Translation, "summary": SessionSummary}
OUT = ROOT / "training" / "data" / "labeled"


class Teacher:
    def __init__(self, url: str, concurrency: int, ollama_model: str | None = None) -> None:
        self.c = httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(600.0))
        self.ollama_model = ollama_model  # si se da, se usa Ollama (/api/generate en crudo) en vez de mlx_lm.server
        self.sem = asyncio.Semaphore(concurrency)
        self.tokens = 0

    async def complete(self, prompt: str, temperature: float, max_tokens: int = 700) -> str:
        async with self.sem:
            for attempt in range(4):
                try:
                    if self.ollama_model:
                        r = await self.c.post("/api/generate", json={
                            "model": self.ollama_model, "prompt": prompt, "raw": True, "stream": False,
                            "options": {"temperature": temperature, "top_p": 0.95, "num_predict": max_tokens,
                                        "stop": [END], "num_ctx": 8192}})
                        r.raise_for_status()
                        d = r.json()
                        self.tokens += d.get("eval_count", 0)
                        return d["response"].split(END)[0].strip()
                    r = await self.c.post("/v1/completions", json={"prompt": prompt, "max_tokens": max_tokens,
                                                                   "temperature": temperature, "top_p": 0.95,
                                                                   "stop": [END]})
                    r.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(30 * (attempt + 1))
            d = r.json()
            self.tokens += d.get("usage", {}).get("completion_tokens", 0)
            return d["choices"][0]["text"].split(END)[0].strip()


def raw_of(rec: dict) -> dict:
    """Entradas originales (las que manda la app), para evaluar después cualquier modelo por el camino de la app."""
    return {k: rec[k] for k in ("task", "system", "user", "messages", "schema") if k in rec}


def key_of(raw: dict) -> str:
    return json.dumps(raw, sort_keys=True, ensure_ascii=False)


def compact(model) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False, separators=(", ", ": "))


def parse_prompt(rec: dict) -> str:
    """Prompt completo: el que ve el maestro."""
    return render_prompt(parse_messages(rec["system"], rec["user"], SCHEMAS[rec["task"]]))


def student_parse_prompt(rec: dict) -> str:
    """Prompt compacto: el que verá el modelo afinado (compact.py)."""
    return render_prompt(parse_messages(compact_parse_system(rec["system"]), rec["user"], SCHEMAS[rec["task"]]))


def student_chat_prompt(rec: dict) -> str:
    return render_prompt(chat_messages(compact_tutor_system(rec["system"]), compact_chat(rec["messages"])))


# ------------------------------------------------------------------------------------------------ eval
def clean_eval_errors(found: ErrorList, learner_text: str, known: list[str]) -> list:
    recurring = {k.split(" -> ")[0].lower() for k in known}
    out = []
    for raw in found.errors:
        if _LEAK.search(raw.corrected) or _LEAK.search(raw.original):
            continue
        for e in narrow_error(raw):
            if not plausible_error(e, learner_text):
                continue
            if any(_overlaps(e.original, x.original) for x in out):
                continue
            if e.category in ("word_choice", "naturalness", "vocabulary") and e.original.lower() not in recurring:
                e = e.model_copy(update={"severity": "minor"})
            out.append(fix_explanation(e))
    return out


async def label_eval(t: Teacher, rec: dict) -> tuple[dict | None, list]:
    u = json.loads(rec["user"])
    prompt = parse_prompt(rec)
    try:
        found = ErrorList.model_validate_json(extract_json(await t.complete(prompt, 0.1)))
    except Exception:  # noqa: BLE001
        return None, []
    errors = clean_eval_errors(found, u["learner_said"], u.get("known_recurring_mistakes", []))
    target = ErrorList(errors=errors)
    return {"task": "eval", "prompt": student_parse_prompt(rec), "completion": compact(target),
            "raw": raw_of(rec)}, errors


def verified_errors(llm_errors: list, learner_text: str, plan: LessonPlan) -> list[dict]:
    rules = rule_based_evaluation(learner_text, plan)
    merged = upgrade_unit_errors(merge_evaluations(
        TurnEvaluation(errors=llm_errors, complexity=rules.complexity, understood_question=True), rules,
        learner_text), plan)
    return [e.model_dump() for e in merged.errors]


# ------------------------------------------------------------------------------------------------ coach
def coach_ok(fb: TurnFeedback, learner_text: str, errors: list[dict]) -> bool:
    if not is_spanish(fb.analysis_es) or len(fb.analysis_es.split()) > 40:
        return False
    corr = [s for s in fb.suggestions if s.kind == "correction"]
    if len(corr) != len(errors) or len(fb.suggestions) > 3:
        return False
    for e in errors:
        if not any(_overlaps(s.you_said, e["original"]) and _overlaps(s.better, e["corrected"]) for s in corr):
            return False
    for s in fb.suggestions:
        if not (is_spanish(s.why_es) and is_spanish(s.how_es)) or not s.example.strip():
            return False
        if s.example.strip().lower() == s.better.strip().lower() or len(s.why_es.split()) > 40:
            return False
    if not (is_spanish(fb.model_answer_why_es) and is_spanish(fb.model_answer_how_es)):
        return False
    cleaned = clean_feedback(fb, learner_text, errors)
    # la app no debe tener que quitar ni añadir nada (salvo convertir la respuesta modelo en sugerencia)
    return len([s for s in cleaned.suggestions if s.kind != "richer_answer"]) == len(fb.suggestions)


async def label_coach(t: Teacher, rec: dict, errors: list[dict]) -> dict | None:
    u = json.loads(rec["user"])
    u["verified_errors"] = [{"you_said": e["original"], "correct": e["corrected"], "reason_es": e["explanation_es"]}
                            for e in errors]
    rec = {**rec, "user": json.dumps(u, ensure_ascii=False)}
    prompt = parse_prompt(rec)
    for temp in (0.2, 0.7):
        try:
            fb = TurnFeedback.model_validate_json(extract_json(await t.complete(prompt, temp, 900)))
        except Exception:  # noqa: BLE001
            continue
        if coach_ok(fb, u["learner_answered"], errors):
            return {"task": "coach", "prompt": student_parse_prompt(rec), "completion": compact(fb),
                    "raw": raw_of(rec)}
    return None


# ------------------------------------------------------------------------------------------------ chat
_SPANISH_OUT = re.compile(r"[¿¡]|\b(que|para|con|pero|muy|bien|cómo|está|también|porque|puedes|dime)\b", re.I)
_QUESTION = re.compile(r"[^.!?]*\?")
_BAD_PRAISE = re.compile(r"\b(perfect|perfectly|flawless|exactly right)\b", re.I)


def chat_score(out: str, rec: dict) -> float:
    """Más alto = mejor. Penaliza lo que el prompt del tutor prohíbe y que los filtros tendrían que quitar."""
    if not out.strip():
        return -99
    notes = " ".join(m["content"] for m in rec["messages"] if m["role"] == "system")
    last_note = next((m["content"] for m in reversed(rec["messages"]) if m["role"] == "system"), "")
    phase = (re.search(r"phase=(\w+)", last_note) or [None, ""])[1]
    spoken = re.sub(r"<(vocab|elicit|fix)\b[^>]*/?>", "", out)
    english = re.sub(r"<es>.*?</es>", " ", spoken, flags=re.S)
    words = len(visible_text(spoken).split())
    score = 0.0
    if re.search(r"[*#`]|^\s*[-•]\s|[\U0001F300-\U0001FAFF]", out, re.M):
        score -= 3
    if _SPANISH_OUT.search(english):
        score -= 3  # español fuera de <es>
    es_spans = re.findall(r"<es>(.*?)</es>", out, re.S)
    if len(es_spans) > (2 if phase == "teach" else 1):
        score -= 1.5 * (len(es_spans) - (2 if phase == "teach" else 1))
    en_words = {"the", "is", "are", "you", "it", "because", "with", "we", "say", "correct", "this", "and", "of", "to"}
    for sp in es_spans:  # <es> con frases en inglés: la voz española las leería mal
        ws = re.findall(r"[a-záéíóúñ]+", re.sub(r"'[^']*'|\"[^\"]*\"", "", sp.lower()))
        if ws and sum(w in en_words for w in ws) / len(ws) > 0.3:
            score -= 2
    learner = next((m["content"] for m in reversed(rec["messages"]) if m["role"] == "user"), "")
    lost = bool(re.search(r"(don'?t|not|no) (understand|entiendo)|what (do )?you mean|repeat|no entiendo",
                          learner + " " + last_note, re.I))
    if any(s.strip().endswith("?") for s in es_spans) and not lost:
        score -= 2  # traduce su pregunta
    limit = {"teach": 95, "wrapup": 85, "review": 70}.get(phase, 60)
    if words > limit:
        score -= (words - limit) / 10
    if words < 4:
        score -= 2
    qs = [q.strip() for q in _QUESTION.findall(english) if len(q.split()) >= 4]
    if phase == "wrapup" and qs:
        score -= 3
    if phase in ("produce", "warmup", "review") and len(qs) > 1:
        score -= 2 * (len(qs) - 1)
    if _BAD_PRAISE.search(english):
        score -= 2
    if ("has a mistake" in last_note or "still has a mistake" in last_note) and re.search(
            r"^\W*(great|excellent|correct|that's right|well done|good job)\b", english.strip(), re.I):
        score -= 3
    prev_q = {q.strip().lower() for m in rec["messages"] if m["role"] == "assistant"
              for q in _QUESTION.findall(re.sub(r"<[^>]+>", "", m["content"]))}
    if any(q.lower() in prev_q for q in qs):
        score -= 2  # vuelve a hacer una pregunta ya hecha
    for o, c in re.findall(r'<fix original="([^"]*)" corrected="([^"]*)"', out):
        if o.strip().lower() == c.strip().lower():
            score -= 3
    asked = re.search(r'(?:ask this new question|give the next exercise|give the learner this task now, clearly):'
                      r' "([^"]+)"', last_note)
    if asked:
        want = set(re.findall(r"[a-z']+", asked.group(1).lower())) - {"the", "a", "to", "you", "your", "and", "is"}
        got = set(re.findall(r"[a-z']+", english.lower()))
        score += 1 if want and len(want & got) >= 0.6 * len(want) else (0 if lost else -1.5)
    if re.search(r"\b(as an ai|system note|the notes?|phase=)", english, re.I):
        score -= 4
    if "[" in out or "]" in out:
        score -= 2
    return score


REGEN_BELOW = 0.0


async def label_chat(t: Teacher, rec: dict, extra: int) -> dict | None:
    msgs = chat_messages(rec["system"], rec["messages"])
    prompt = render_prompt(msgs)
    cands = [rec["output"].strip()]
    if chat_score(cands[0], rec) < REGEN_BELOW:  # la respuesta original ya es buena: no se gasta al maestro
        cands += await asyncio.gather(*[t.complete(prompt, 0.8, 300) for _ in range(extra)])
    scored = sorted(((chat_score(c, rec), -i, c) for i, c in enumerate(cands)), reverse=True)
    best, _, text = scored[0]
    if best < -1.5:
        return None
    return {"task": "chat", "prompt": student_chat_prompt(rec), "completion": text, "score": best,
            "raw": raw_of(rec)}


# ------------------------------------------------------------------------------------------------ resto
async def label_simple(t: Teacher, rec: dict) -> dict | None:
    prompt = parse_prompt(rec)
    schema = SCHEMAS[rec["task"]]
    for temp in (0.2, 0.7):
        try:
            out = schema.model_validate_json(extract_json(await t.complete(prompt, temp, 900)))
        except Exception:  # noqa: BLE001
            continue
        task = rec["task"]
        if task == "facts":
            said = " ".join(json.loads(rec["user"]).get("learner_said", []))
            if not all(grounded(f, said) for f in out.facts) or len(out.facts) > 6:
                continue
        if task == "translate" and not is_spanish(out.es):
            continue
        if task == "help" and not (2 <= len(out.starters) <= 3 and len(out.words) <= 3 and all(
                len(s.split()) <= 11 for s in out.starters)):
            continue
        return {"task": task, "prompt": student_parse_prompt(rec), "completion": compact(out), "raw": raw_of(rec)}
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--chat-extra", type=int, default=1, help="candidatos nuevos por turno del profesor")
    ap.add_argument("--coach-max", type=int, default=600)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--ollama", default=None, help="modelo de Ollama como maestro (p. ej. qwen3.5:9b)")
    ap.add_argument("--resume", action="store_true", help="conserva lo ya etiquetado y sólo hace lo que falta")
    a = ap.parse_args()
    recs = [json.loads(line) for p in a.raw for line in open(p)]
    OUT.mkdir(parents=True, exist_ok=True)
    t = Teacher(a.url, a.concurrency, a.ollama)
    names = ("eval", "coach", "chat", "help", "translate", "facts", "summary")
    done: dict[str, dict[str, dict]] = {k: {} for k in names}
    if a.resume:
        for k in names:
            f = OUT / f"{a.tag}_{k}.jsonl"
            for line in (f.open() if f.exists() else []):
                x = json.loads(line)
                done[k][key_of(x["raw"])] = x
        print("ya hechos:", {k: len(v) for k, v in done.items()}, flush=True)
    files = {k: (OUT / f"{a.tag}_{k}.jsonl").open("a" if a.resume else "w") for k in names}
    stats: dict[str, list[int]] = {}

    def save(task: str, item: dict | None, rec: dict) -> None:
        st = stats.setdefault(task, [0, 0])
        st[1] += 1
        if item:
            item["learner"] = rec.get("learner", "")
            st[0] += 1
            files[task].write(json.dumps(item, ensure_ascii=False) + "\n")
            files[task].flush()

    rng = random.Random(0)
    coach_recs = [r for r in recs if r["task"] == "coach"]
    coach_keep = set(map(id, rng.sample(coach_recs, min(a.coach_max, len(coach_recs)))))

    async def per_eval(rec: dict, coach: dict | None) -> None:
        prev = done["eval"].get(key_of(raw_of(rec)))
        if prev is not None:
            item, errors = prev, ErrorList.model_validate_json(prev["completion"]).errors
        else:
            item, errors = await label_eval(t, rec)
            save("eval", item, rec)
        if coach is not None and key_of(raw_of(coach)) in done["coach"]:
            return
        if coach is not None and item is not None and id(coach) in coach_keep:
            u = json.loads(rec["user"])
            plan = LessonPlan.from_dict(rec["plan"])
            save("coach", await label_coach(t, coach, verified_errors(errors, u["learner_said"], plan)), rec)

    jobs = []
    for i, rec in enumerate(recs):
        if rec["task"] != "eval" and key_of(raw_of(rec)) in done.get(rec["task"], {}):
            continue
        if rec["task"] == "eval":
            said = json.loads(rec["user"])["learner_said"]
            coach = next((c for c in recs[i + 1:i + 8] if c["task"] == "coach" and
                          json.loads(c["user"])["learner_answered"] == said), None)
            jobs.append(per_eval(rec, coach))
        elif rec["task"] == "chat":
            async def one_chat(r=rec):
                save("chat", await label_chat(t, r, a.chat_extra), r)
            jobs.append(one_chat())
        elif rec["task"] in ("help", "translate", "facts", "summary"):
            async def one(r=rec):
                save(r["task"], await label_simple(t, r), r)
            jobs.append(one())

    n_done = 0

    async def wrap(j):
        nonlocal n_done
        try:
            await j
        except Exception as e:  # noqa: BLE001  un fallo suelto no tira la tanda entera
            print("fallo:", repr(e)[:200], flush=True)
        n_done += 1
        if n_done % 25 == 0:
            print(f"{n_done}/{len(jobs)} · tokens {t.tokens} · " + " ".join(f"{k} {v[0]}/{v[1]}"
                                                                       for k, v in stats.items()), flush=True)

    await asyncio.gather(*[wrap(j) for j in jobs])
    print("FIN", {k: f"{v[0]}/{v[1]}" for k, v in stats.items()}, "tokens", t.tokens)


if __name__ == "__main__":
    asyncio.run(main())
