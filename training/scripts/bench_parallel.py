import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import httpx
from tutor.ai.prompts import EVALUATOR_SYSTEM
from tutor.ai.schemas import ErrorList
from tutor.learning.curriculum import GRAMMAR
S = ["um my sister go to the beach on the weekend and she don't work, she is nurse", "yesterday I go to the cinema with my wife and we see a very good film",
     "I am agree with you, I go always to work by bike", "my father have 60 years and he live in Toledo", "I have went to London two years ago",
     "she work in a bank and she don't like it", "we was very tired after the trip", "I didn't went to the party because I was sick"]
async def one(c, model, said):
    body = {"model": model, "stream": False, "think": False, "keep_alive": "10m", "format": ErrorList.model_json_schema(),
            "options": {"num_ctx": 8192, "temperature": 0.1},
            "messages": [{"role": "system", "content": EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)) + "\nAnswer ONLY with JSON that matches the schema."},
                         {"role": "user", "content": json.dumps({"tutor_said": "Tell me about it.", "learner_said": said, "target_vocabulary": [], "target_grammar": ["present_simple"], "todays_structure": "he/she/it works", "known_recurring_mistakes": []})}]}
    r = (await c.post("http://127.0.0.1:11434/api/chat", json=body)).json()
    return r["eval_count"]
async def main():
    model, n = sys.argv[1], int(sys.argv[2])
    async with httpx.AsyncClient(timeout=900) as c:
        await one(c, model, S[0])
        t = time.time(); toks = await asyncio.gather(*[one(c, model, S[i % 8]) for i in range(n)])
    print(model, n, f"{time.time()-t:.1f}s", "agg tok/s", round(sum(toks)/(time.time()-t), 1))
asyncio.run(main())
