# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Stage-1b: bake ``<effect_0..C-1>`` special tokens into an Action checkpoint.

These tokens are *appended after* the existing FAST ``<robot_action_*>`` tokens,
so the resulting model emits a latent plan (effect tokens) followed by FAST
action tokens. The new id range is printed and written to
``effect_token_id_range.json`` — wire it into ``QwenEffect``'s config.

Example:
    python examples/ActionEffect/train_files/add_effect_tokens.py \
        --model-id  /path/Qwen3-VL-4B-Instruct-Action \
        --save-dir  /path/Qwen3-VL-4B-Instruct-ActionEffect \
        --tokenizer-ckpt /path/runs/effect_tok/checkpoints/effect_tokenizer.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADD_TOOL = _REPO_ROOT / "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py"


def _load_add_tool():
    spec = importlib.util.spec_from_file_location("_add_qwen_special_tokens", _ADD_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vocab_from_ckpt(ckpt_path: str):
    """Return (total_vocab, num_codes, num_quantizers). With RVQ the flat vocab is
    ``num_quantizers * num_codes`` (token id = level*num_codes + code)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    num_codes = int(cfg.get("num_codes", 512))
    num_quantizers = int(cfg.get("num_quantizers", 1))
    return num_quantizers * num_codes, num_codes, num_quantizers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="The *-Action checkpoint (already has FAST tokens)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--tokenizer-ckpt", required=True, help="Stage-1 effect_tokenizer.pt (to read num_codes)")
    p.add_argument("--init-strategy", default="normal", choices=["avg", "normal", "zero"])
    args = p.parse_args()

    total_vocab, num_codes, num_quantizers = _vocab_from_ckpt(args.tokenizer_ckpt)
    effect_tokens = [f"<effect_{i}>" for i in range(total_vocab)]
    print(f"[add_effect_tokens] adding {total_vocab} effect tokens "
          f"(num_codes={num_codes} x num_quantizers={num_quantizers}) to {args.model_id}")

    tool = _load_add_tool()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, attn_implementation="sdpa", dtype=torch.bfloat16
    )

    mapping, added, start_idx, end_idx = tool.add_new_tokens(
        model=model, tokenizer=tokenizer, new_tokens=effect_tokens, init_strategy=args.init_strategy, as_special=True
    )
    tool.save_bundle(model, tokenizer, mapping, args.save_dir, processor_src=args.model_id, padding_side="left")
    tool.reload_and_check(args.save_dir, effect_tokens[:8])

    rng = {
        "effect_token_min": int(tokenizer.convert_tokens_to_ids("<effect_0>")),
        "effect_token_max": int(tokenizer.convert_tokens_to_ids(f"<effect_{total_vocab - 1}>")),
        "num_codes": num_codes,
        "num_quantizers": num_quantizers,
        "total_effect_vocab": total_vocab,
        "added_now": added,
    }
    with open(os.path.join(args.save_dir, "effect_token_id_range.json"), "w") as f:
        json.dump(rng, f, indent=2)
    print(f"[add_effect_tokens] effect token id range: {rng}")


if __name__ == "__main__":
    main()
