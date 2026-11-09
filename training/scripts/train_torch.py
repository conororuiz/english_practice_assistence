"""Entrenamiento LoRA en GPU NVIDIA (PyTorch + PEFT): la misma receta que train.py (MLX), para el PC Linux.

- Sólo el modelo de texto (Qwen3_5ForCausalLM); la parte de visión no se toca y se conserva al fusionar.
- Pérdida sólo en la respuesta: la cabeza (vocabulario de 248k) se aplica únicamente a esas posiciones.
- LoRA rango 16 / alpha 32 en atención, atención lineal y MLP de todas las capas; bf16; grad checkpointing.
- Guarda el mejor adaptador (pérdida de validación) en formato PEFT en --out.

    training/.venv/bin/python training/scripts/train_torch.py --base training/base/Qwen3.5-4B \
        --out training/adapters/tutor-sam-4b
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "training" / "data" / "sft"
END = "<|im_end|>"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def load_split(tok, name: str, max_len: int) -> list[tuple[list[int], int]]:
    out, skipped = [], 0
    for line in open(SFT / f"{name}.jsonl"):
        x = json.loads(line)
        p = tok.encode(x["prompt"], add_special_tokens=False)
        c = tok.encode(x["completion"] + END, add_special_tokens=False)
        if len(p) + len(c) > max_len:
            skipped += 1
            continue
        out.append((p + c, len(p)))
    print(f"{name}: {len(out)} ejemplos ({skipped} descartados por longitud)", flush=True)
    return out


def completion_loss(model, tokens: list[int], offset: int, device) -> tuple[torch.Tensor, int]:
    ids = torch.tensor([tokens], device=device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state  # (1, T, H)
    h = hidden[0, offset - 1:-1]  # posiciones que predicen la respuesta
    logits = base.lm_head(h).float()
    target = ids[0, offset:]
    return F.cross_entropy(logits, target, reduction="sum"), target.numel()


@torch.no_grad()
def evaluate(model, data, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for tokens, off in data:
        loss, k = completion_loss(model, tokens, off, device)
        tot += loss.item()
        n += k
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--iters", type=int, default=0, help="limita los pasos (pruebas)")
    ap.add_argument("--report", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train = load_split(tok, "train", a.max_len)
    valid = load_split(tok, "valid", a.max_len)
    steps = a.iters or math.ceil(len(train) * a.epochs / a.accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, steps // 20)

    def lr_at(s: int) -> float:
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w")
    best = evaluate(model, valid, device)
    print(f"valid inicial {best:.4f}", flush=True)
    best = float("inf")
    eval_every = max(50, steps // 6)
    order: list[int] = []
    t0, ntok, run_loss, run_n = time.time(), 0, 0.0, 0
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        acc_loss, acc_n = 0.0, 0
        batch = []
        for _ in range(a.accum):
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
            batch.append(train[order.pop()])
        total = sum(len(t) - o for t, o in batch)
        for tokens, off in batch:
            loss, k = completion_loss(model, tokens, off, device)
            (loss / total).backward()
            acc_loss += loss.item()
            acc_n += k
            ntok += len(tokens)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run_loss += acc_loss
        run_n += acc_n
        if (step + 1) % a.report == 0:
            dt = time.time() - t0
            rec = {"step": step + 1, "loss": run_loss / run_n, "lr": lr_at(step), "tok_s": ntok / dt,
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss, run_n = 0.0, 0
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            v = evaluate(model, valid, device)
            rec = {"step": step + 1, "valid": v}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if v < best:
                best = v
                model.save_pretrained(out)
                print(f"guardado (valid {v:.4f})", flush=True)
    print(f"FIN mejor valid {best:.4f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
