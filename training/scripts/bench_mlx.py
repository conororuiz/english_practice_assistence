import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler
from tutor.ai.prompts import EVALUATOR_SYSTEM
from tutor.learning.curriculum import GRAMMAR
S = ["um my sister go to the beach on the weekend and she don't work, she is nurse", "yesterday I go to the cinema with my wife and we see a very good film",
     "I am agree with you, I go always to work by bike", "my father have 60 years and he live in Toledo", "I have went to London two years ago",
     "she work in a bank and she don't like it", "we was very tired after the trip", "I didn't went to the party because I was sick"]
model, tok = load(sys.argv[1])
def prompt(said):
    msgs = [{"role": "system", "content": EVALUATOR_SYSTEM.format(skills=", ".join(GRAMMAR)) + "\nAnswer ONLY with JSON: {\"errors\": [{\"original\", \"corrected\", \"category\", \"skill_key\", \"severity\", \"explanation_es\"}]}"},
            {"role": "user", "content": json.dumps({"tutor_said": "Tell me about it.", "learner_said": said, "target_vocabulary": [], "target_grammar": ["present_simple"], "todays_structure": "he/she/it works", "known_recurring_mistakes": []})}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
for n in [int(x) for x in sys.argv[2:]]:
    ps = [prompt(S[i % 8]) for i in range(n)]
    t = time.time()
    r = batch_generate(model, tok, ps, max_tokens=500, sampler=make_sampler(temp=0.1))
    dt = time.time() - t
    toks = sum(len(tok.encode(x)) for x in r.texts)
    print(n, f"{dt:.1f}s agg tok/s {toks/dt:.1f}  peak GB {mx.get_peak_memory()/1e9:.1f}", flush=True)
print(r.texts[0][:700]); print(r.texts[3][:500])
