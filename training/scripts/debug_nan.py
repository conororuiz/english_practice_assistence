import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_delta import install
install()
import mlx.core as mx
from mlx_lm import load
import train as T
model, tok = load(sys.argv[1])
x = json.loads(open(Path(__file__).resolve().parents[1] / "data/sft/train.jsonl").readline())
p = tok.encode(x["prompt"], add_special_tokens=False); c = tok.encode(x["completion"] + "<|im_end|>", add_special_tokens=False)
toks = mx.array(p + c)[None]
model.eval(); l1, _ = T.completion_loss(model, toks, len(p)); print("eval", l1.item())
model.train(); l2, _ = T.completion_loss(model, toks, len(p)); print("train", l2.item())
# localizar: salidas de la capa lineal con kernel vs chunked
from mlx_lm.models import qwen3_5
import chunked_delta as cd
orig = cd.chunk_gated_delta
def wrapped(q, k, v, g_log, beta, state, chunk=64):
    y, s = orig(q, k, v, g_log, beta, state, chunk)
    print("layer: q", mx.abs(q).max().item(), "k", mx.abs(k).max().item(), "g_log min", g_log.min().item(), "beta", beta.max().item(), "y nan", mx.isnan(y).any().item(), "ymax", mx.abs(y).max().item())
    return y, s
cd.chunk_gated_delta = wrapped
model.train(); l2, _ = T.completion_loss(model, toks, len(p)); print("train", l2.item())
