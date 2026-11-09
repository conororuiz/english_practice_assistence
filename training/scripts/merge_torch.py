"""Fusiona un adaptador PEFT (train_torch.py) en los pesos HF originales, sin GPU ni MLX (PC Linux).
Conserva todo lo demás (visión, MTP, tokenizer, config), igual que merge.py.

    training/.venv/bin/python training/scripts/merge_torch.py --base training/base/Qwen3.5-4B \
        --adapter training/adapters/tutor-sam-4b --out training/merged/tutor-sam
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out, ad = Path(a.base), Path(a.out), Path(a.adapter)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(ad / "adapter_model.safetensors"))
    deltas: dict[str, torch.Tensor] = {}
    for k, A in lora.items():
        if not k.endswith(".lora_A.weight"):
            continue
        prefix = k[: -len(".lora_A.weight")]
        B = lora[prefix + ".lora_B.weight"]
        assert prefix.startswith("base_model.model.model."), prefix
        hf = "model.language_model." + prefix[len("base_model.model.model."):] + ".weight"
        deltas[hf] = scale * (B.float() @ A.float())  # [out, in]
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    used = 0
    for shard in sorted(base.glob("*.safetensors")):
        w = load_file(str(shard))
        for name, d in deltas.items():
            if name in w:
                assert w[name].shape == d.shape, (name, w[name].shape, d.shape)
                w[name] = (w[name].float() + d).to(w[name].dtype)
                used += 1
        save_file(w, str(out / shard.name), metadata={"format": "pt"})
        del w
    assert used == len(deltas), f"sólo se aplicaron {used} de {len(deltas)} deltas"
    print(f"{used} matrices fusionadas → {out}")


if __name__ == "__main__":
    main()
