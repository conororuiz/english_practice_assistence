import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import httpx
from tutor.ai.prompts import EVALUATOR_SYSTEM
from tutor.ai.schemas import TurnEvaluation
from tutor.learning.curriculum import GRAMMAR

async def run(model, think):
    sys_ = EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)) + "\nAnswer ONLY with JSON that matches the schema."
    user = json.dumps({"tutor_said": "What does your sister do on weekends?", "learner_said": "um my sister go to the beach on the weekend and she don't work, she is nurse",
        "target_vocabulary": ["usually", "weekend"], "target_grammar": ["present_simple"], "todays_structure": "he/she/it works", "known_recurring_mistakes": []})
    body = {"model": model, "stream": False, "think": think, "keep_alive": "10m", 
            "options": {"num_ctx": 8192, "temperature": 0.1},
            "messages": [{"role": "system", "content": sys_}, {"role": "user", "content": user}]}
    if not model.startswith("tutor"): body["format"] = TurnEvaluation.model_json_schema()
    async with httpx.AsyncClient(timeout=600) as c:
        t = time.time(); r = (await c.post("http://127.0.0.1:11434/api/chat", json=body)).json()
    m = r["message"]
    print(model, "think" if think else "nothink", f"{time.time()-t:.1f}s", "gen tok/s", round(r["eval_count"]/(r["eval_duration"]/1e9),1),
          "prompt tok", r["prompt_eval_count"], "thinking chars", len(m.get("thinking") or ""))
    print("  ", m["content"][:600])

async def main():
    for model, think in [(m, False) for m in sys.argv[1:]]:
        await run(model, think)
asyncio.run(main())
