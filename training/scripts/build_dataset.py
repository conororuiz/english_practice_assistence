"""Junta los datos etiquetados (relabel.py) y los ejemplos gold escritos a mano en train/valid/test.

- test: los alumnos simulados indicados con --test-learners (nunca se entrenan con ellos).
- gold: se repiten --gold-repeat veces (son pocos y los más fiables).
- Mezcla por tarea: el chat es lo más importante (la voz del profesor); se recorta lo que sobre de cada tarea.

    .venv/bin/python training/scripts/build_dataset.py --test-learners r1-3 r1-9
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "training"))

from fmt import render_prompt  # noqa: E402
from gold.coach_gold import GOLD as COACH_GOLD  # noqa: E402
from gold.eval_gold import GOLD as EVAL_GOLD  # noqa: E402
from tutor.ai.coach import COACH_SYSTEM  # noqa: E402
from tutor.ai.compact import TASKS  # noqa: E402
from tutor.ai.prompts import EVALUATOR_SYSTEM  # noqa: E402
from tutor.learning.curriculum import GRAMMAR  # noqa: E402
from tutor.ai.ollama_llm import parse_messages  # noqa: E402
from tutor.ai.schemas import DetectedError, ErrorList, TurnFeedback  # noqa: E402
from tutor.learning.route import ROUTE  # noqa: E402

LAB = ROOT / "training" / "data" / "labeled"
OUT = ROOT / "training" / "data" / "sft"


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def gold_eval(rng: random.Random) -> list[dict]:
    out = []
    for i, (unit, tutor, said, errors) in enumerate(EVAL_GOLD):
        u = ROUTE[unit - 1]
        terms = [t for t, _ in u.vocab]
        rng.shuffle(terms)
        user = json.dumps({"tutor_said": tutor, "learner_said": said, "target_vocabulary": terms[:rng.randint(2, 6)],
                           "target_grammar": [u.skill], "todays_structure": u.form,
                           "known_recurring_mistakes": []}, ensure_ascii=False)
        errs = [DetectedError(original=o, corrected=c, category=cat, skill_key=sk, severity=sev, explanation_es=ex)
                for o, c, cat, sk, sev, ex in errors]
        prompt = render_prompt(parse_messages(TASKS["eval"], user, ErrorList))
        raw = {"task": "eval", "system": EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)), "user": user,
               "schema": "ErrorList"}
        out.append({"task": "eval", "prompt": prompt, "completion": compact(ErrorList(errors=errs).model_dump()),
                    "learner": "gold-test" if i % 5 == 4 else "gold", "raw": raw})
    return out


def gold_coach() -> list[dict]:
    out = []
    for i, (unit, tutor, said, errors, fb) in enumerate(COACH_GOLD):
        u = ROUTE[unit - 1]
        user = json.dumps({"teacher_asked": tutor, "learner_answered": said,
                           "verified_errors": [{"you_said": o, "correct": c, "reason_es": r} for o, c, r in errors],
                           "todays_topic": u.title, "todays_structure": u.form,
                           "todays_words": [t for t, _ in u.vocab][:6]}, ensure_ascii=False)
        TurnFeedback.model_validate(fb)
        prompt = render_prompt(parse_messages(TASKS["coach"], user, TurnFeedback))
        raw = {"task": "coach", "system": COACH_SYSTEM, "user": user, "schema": "TurnFeedback"}
        out.append({"task": "coach", "prompt": prompt, "completion": compact(fb),
                    "learner": "gold-test" if i % 6 == 5 else "gold", "raw": raw})
    return out


def gold_chat() -> list[dict]:
    from gold.chat_gold import GOLD as CHAT_GOLD
    from relabel import chat_score, student_chat_prompt

    raw = [json.loads(line) for f in ("r1", "r2") for line in open(ROOT / "training" / "data" / "raw" / f"{f}.jsonl")]
    out = []
    for idx, text in CHAT_GOLD.items():
        rec = raw[idx]
        assert rec["task"] == "chat", idx
        score = chat_score(text, rec)
        if score < -1.5:
            print(f"aviso: gold de chat {idx} puntúa {score}")
        out.append({"task": "chat", "prompt": student_chat_prompt(rec), "completion": text,
                    "learner": "gold", "raw": {k: rec[k] for k in ("task", "system", "messages")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-learners", nargs="*", default=[])
    ap.add_argument("--gold-repeat", type=int, default=3)
    ap.add_argument("--valid-frac", type=float, default=0.04)
    ap.add_argument("--max-per-task", type=json.loads,
                    default='{"chat": 100000, "eval": 1200, "coach": 700, "help": 150, "translate": 150, '
                            '"facts": 150, "summary": 150}')
    a = ap.parse_args()
    rng = random.Random(7)
    items = [json.loads(line) for p in sorted(LAB.glob("*.jsonl")) for line in open(p)]
    test = [x for x in items if x.get("learner") in a.test_learners]
    pool = [x for x in items if x.get("learner") not in a.test_learners]
    by_task: dict[str, list[dict]] = {}
    for x in pool:
        by_task.setdefault(x["task"], []).append(x)
    train, valid = [], []
    for task, xs in by_task.items():
        rng.shuffle(xs)
        xs = xs[:a.max_per_task.get(task, 1000)]
        nv = max(1, int(len(xs) * a.valid_frac))
        valid += xs[:nv]
        train += xs[nv:]
    # Evaluaciones sin errores: que no dominen (el modelo aprendería a decir siempre "sin errores").
    empty = [x for x in train if x["task"] == "eval" and x["completion"] == '{"errors": []}']
    full = [x for x in train if x["task"] == "eval" and x["completion"] != '{"errors": []}']
    if len(empty) > len(full):
        drop = set(map(id, empty[len(full):]))
        train = [x for x in train if id(x) not in drop]
    gold = gold_eval(rng) + gold_coach() + gold_chat()
    test += [g for g in gold if g["learner"] == "gold-test"]  # nunca se entrenan: miden la calidad final
    train += [g for g in gold if g["learner"] == "gold"] * a.gold_repeat
    rng.shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, xs in (("train", train), ("valid", valid), ("test", test)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        counts: dict[str, int] = {}
        for x in xs:
            counts[x["task"]] = counts.get(x["task"], 0) + 1
        print(name, len(xs), counts)


if __name__ == "__main__":
    main()
