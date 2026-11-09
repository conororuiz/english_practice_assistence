import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_delta import install
install()
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
import train as T
model, tok = load(sys.argv[1])
toks = mx.array([list(range(1000, 3048))])
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": 16, "scale": 2.0, "dropout": 0.0, "keys": ["mlp.down_proj", "self_attn.q_proj"]})
def tm(fn, n=3):
    mx.eval(fn()); t = time.time()
    for _ in range(n): mx.eval(fn())
    return (time.time() - t) / n
model.eval(); print("eval fwd", round(tm(lambda: T.completion_loss(model, toks, 2000)[0]), 2))
model.train(); print("train fwd", round(tm(lambda: T.completion_loss(model, toks, 2000)[0]), 2))
lg = nn.value_and_grad(model, T.completion_loss)
print("fwd+bwd", round(tm(lambda: lg(model, toks, 2000)), 2))
grad_checkpoint(model.layers[0])
print("fwd+bwd ckpt", round(tm(lambda: lg(model, toks, 2000)), 2))
