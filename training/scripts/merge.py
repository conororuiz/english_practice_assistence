"""Fusiona el adaptador LoRA (train.py) en los pesos originales de Hugging Face (bf16), conservando todo lo demás
(torre de visión, cabeza MTP, tokenizer, config). El resultado se importa en Ollama con `ollama create -q int4`.

    training/.venv/bin/python training/scripts/merge.py --base training/base/Qwen3.5-4B \
        --adapter training/adapters/tutor-4b --out training/merged/tutor-4b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--weights", default="adapters.safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    deltas: dict[str, mx.array] = {}
    if "peft_type" in cfg:  # adaptador PEFT (train_torch.py, PC con NVIDIA)
        scale = cfg["lora_alpha"] / cfg["r"]
        lora = mx.load(str(Path(a.adapter) / "adapter_model.safetensors"))
        for k in lora:
            if not k.endswith(".lora_A.weight"):
                continue
            prefix = k[: -len(".lora_A.weight")]
            A = lora[k].astype(mx.float32)  # [r, in]
            B = lora[prefix + ".lora_B.weight"].astype(mx.float32)  # [out, r]
            # peft: base_model.model.model.layers.N.x  →  HF: model.language_model.layers.N.x.weight
            assert prefix.startswith("base_model.model.model."), prefix
            hf = "model.language_model." + prefix[len("base_model.model.model."):] + ".weight"
            deltas[hf] = scale * (B @ A)
        lora = {}
    else:
        scale = cfg["lora_parameters"]["scale"]
        lora = mx.load(str(Path(a.adapter) / a.weights))
    for k in lora:
        if not k.endswith(".lora_a"):
            continue
        prefix = k[: -len(".lora_a")]
        A, B = lora[k].astype(mx.float32), lora[prefix + ".lora_b"].astype(mx.float32)
        # mlx: language_model.model.layers.N.x  →  HF: model.language_model.layers.N.x.weight
        assert prefix.startswith("language_model.model."), prefix
        hf = "model.language_model." + prefix[len("language_model.model."):] + ".weight"
        deltas[hf] = (scale * (A @ B)).T  # [out, in]
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):
            shutil.copy2(f, out / f.name)
    used = 0
    for shard in sorted(base.glob("*.safetensors")):
        w = mx.load(str(shard))
        for name, d in deltas.items():
            if name in w:
                assert w[name].shape == d.shape, (name, w[name].shape, d.shape)
                w[name] = (w[name].astype(mx.float32) + d).astype(w[name].dtype)
                used += 1
        mx.save_safetensors(str(out / shard.name), w, metadata={"format": "pt"})
        del w
    assert used == len(deltas), f"sólo se aplicaron {used} de {len(deltas)} deltas"
    print(f"{used} matrices fusionadas → {out}")


if __name__ == "__main__":
    main()
