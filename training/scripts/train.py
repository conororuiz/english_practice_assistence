"""Fase 3: LoRA sobre Qwen3.5 con mlx-lm, con el formato exacto de fmt.py y pérdida sólo en la respuesta.

    training/.venv/bin/python training/scripts/train.py --base training/base/Qwen3.5-4B --out training/adapters/tutor-4b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx.utils import tree_flatten, tree_map
import numpy as np
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_delta import install  # noqa: E402

install()  # GatedDeltaNet por bloques en entrenamiento (memoria O(T/64) en vez de O(T))
ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "training" / "data" / "sft"


def load_split(tok, name: str, max_len: int) -> list[tuple[list[int], int]]:
    out, skipped = [], 0
    for line in open(SFT / f"{name}.jsonl"):
        x = json.loads(line)
        p = tok.encode(x["prompt"], add_special_tokens=False)
        c = tok.encode(x["completion"] + "<|im_end|>", add_special_tokens=False)
        if len(p) + len(c) > max_len:
            skipped += 1
            continue
        out.append((p + c, len(p)))
    print(f"{name}: {len(out)} ejemplos ({skipped} demasiado largos)")
    return out


def completion_loss(model, tokens, offset: int):
    """Pérdida sólo en la respuesta, y el lm_head (vocabulario de 248k) sólo sobre esas posiciones: con prompts de
    ~3000 tokens, los logits completos no caben en memoria."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    lm = model.language_model
    hidden = lm.model(inputs)[:, offset - 1:]
    logits = lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets[:, offset - 1:])
    return ce.mean(), ce.shape[1]


def evaluate(model, data) -> float:
    total, n = 0.0, 0
    model.eval()
    for tokens, off in data:
        loss, k = completion_loss(model, mx.array(tokens)[None], off)
        total += loss.item() * k
        n += k
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--layers", type=int, default=-1, help="capas con LoRA (-1 = todas)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--report", type=int, default=20)
    ap.add_argument("--iters", type=int, default=0, help="fija las iteraciones (0 = según epochs)")
    a = ap.parse_args()

    mx.random.seed(0)
    model, tok = load(a.base)
    train_set = load_split(tok, "train", a.max_len)
    valid_set = load_split(tok, "valid", a.max_len)
    model.freeze()
    keys = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "linear_attn.in_proj_qkvz", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    present = {n.split(".", 2)[-1] for n, _ in model.layers[0].named_modules()} | \
              {n for layer in model.layers[:4] for n, _ in layer.named_modules()}
    keys = {k for k in keys if k in present}
    print("LoRA en:", sorted(keys))
    n_layers = len(model.layers) if a.layers < 0 else a.layers
    linear_to_lora_layers(model, n_layers, {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": 0.05,
                                            "keys": sorted(keys)})
    print_trainable_parameters(model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": n_layers, "base": str(a.base),
        "lora_parameters": {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": 0.05, "keys": sorted(keys)}},
        indent=2))
    iters = a.iters or math.ceil(len(train_set) * a.epochs / a.batch)
    warm = min(max(10, iters // 30), max(1, iters // 2))
    sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, warm),
                                  optim.cosine_decay(a.lr, iters - warm, a.lr * 0.05)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    grad_checkpoint(model.layers[0])
    log = (out / "log.jsonl").open("w")
    loss_and_grad = nn.value_and_grad(model, completion_loss)
    order = []
    rng = np.random.default_rng(0)
    while len(order) < iters:
        order += list(rng.permutation(len(train_set)))
    acc, acc_n, run_loss, run_tok, t0 = None, 0, 0.0, 0, time.time()
    eval_every = max(100, iters // 6)
    best = float("inf")
    print(f"{iters} iteraciones, acumulación {a.accum}; val inicial {evaluate(model, valid_set):.3f}", flush=True)
    for it in range(1, iters + 1):
        tokens, off = train_set[order[it - 1]]
        (loss, k), grads = loss_and_grad(model, mx.array(tokens)[None], off)
        acc = grads if acc is None else tree_map(lambda x, y: x + y, acc, grads)
        acc_n += 1
        mx.eval(loss, acc)
        run_loss += loss.item()
        run_tok += len(tokens)
        if acc_n == a.accum or it == iters:
            opt.update(model, tree_map(lambda g: g / acc_n, acc))
            mx.eval(model.parameters(), opt.state)
            acc, acc_n = None, 0
        if it % a.report == 0:
            info = {"it": it, "loss": round(run_loss / a.report, 4), "tok_s": round(run_tok / (time.time() - t0)),
                    "lr": float(opt.learning_rate), "peak_gb": round(mx.get_peak_memory() / 1e9, 1)}
            print(info, flush=True)
            log.write(json.dumps(info) + "\n"); log.flush()
            run_loss, run_tok, t0 = 0.0, 0, time.time()
        if it % eval_every == 0 or it == iters:
            v = evaluate(model, valid_set)
            print({"it": it, "val": round(v, 4)}, flush=True)
            log.write(json.dumps({"it": it, "val": v}) + "\n"); log.flush()
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(out / f"{it:06d}_adapters.safetensors"), weights)
            if v < best:
                best = v
                mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    print("mejor val", round(best, 4))


if __name__ == "__main__":
    main()
