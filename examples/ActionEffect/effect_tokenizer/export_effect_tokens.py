# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Diagnostics: codebook-usage statistics for a trained effect tokenizer.

Runs the frozen Stage-1 encoder over (a subset of) the dataset and reports how
many codes are actually used and their frequency. Useful to sanity-check the VQ
bottleneck (avoid codebook collapse) before committing to a vocabulary size.

Example:
    python examples/ActionEffect/effect_tokenizer/export_effect_tokens.py \
        --data_root_dir /path/LIBERO --data_mix libero_all_effect \
        --tokenizer-ckpt /path/effect_tokenizer.pt --max_batches 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from examples.ActionEffect.effect_tokenizer.dataset import build_effect_dataset, effect_collate
from examples.ActionEffect.effect_tokenizer.model import EffectAwareTokenizer, EffectTokenizerConfig


def load_tokenizer(ckpt_path: str, device) -> EffectAwareTokenizer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = EffectTokenizerConfig(**ckpt["config"])
    model = EffectAwareTokenizer(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root_dir", required=True)
    p.add_argument("--data_mix", default="libero_all_effect")
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_tokenizer(args.tokenizer_ckpt, device)

    dataset = build_effect_dataset(args.data_root_dir, args.data_mix, {"video_backend": "torchvision_av"})
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=effect_collate)

    counters: list[Counter] | None = None
    n = 0
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        codes = model.encode_to_codes(
            batch["obs"].to(device), batch["future_obs"].to(device),
            batch["action"].to(device), batch["lang"],
        )  # (B, N, L)
        if codes.ndim == 2:
            codes = codes.unsqueeze(-1)
        if counters is None:
            counters = [Counter() for _ in range(codes.shape[-1])]
        for level in range(codes.shape[-1]):
            for c in codes[..., level].reshape(-1).tolist():
                counters[level][int(c)] += 1
        n += codes.numel()

    counters = counters or [Counter()]
    per_level = []
    for level, counter in enumerate(counters):
        used = len(counter)
        per_level.append({
            "level": level,
            "used_codes": used,
            "usage_ratio": used / model.cfg.num_codes,
            "top20": counter.most_common(20),
        })
    stats = {
        "num_codes": model.cfg.num_codes,
        "num_quantizers": model.cfg.num_quantizers,
        "total_effect_vocab": model.total_effect_vocab,
        "total_tokens_counted": n,
        "per_level": per_level,
    }
    printable = {
        **{k: v for k, v in stats.items() if k != "per_level"},
        "per_level": [{k: v for k, v in item.items() if k != "top20"} for item in per_level],
    }
    print(json.dumps(printable, indent=2))
    for item in per_level:
        print(f"level {item['level']} top-20 codes:", item["top20"])
    out = args.out or str(Path(args.tokenizer_ckpt).with_name("effect_token_stats.json"))
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
