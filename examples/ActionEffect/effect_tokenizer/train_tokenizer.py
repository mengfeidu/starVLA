# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Stage-1 trainer for the action-first Effect Tokenizer.

Standalone (does not go through ``train_starvla``) so the whole Stage-1 logic
lives in this folder. Compatible with ``accelerate launch`` (single or multi-GPU).

Example:
    accelerate launch examples/ActionEffect/effect_tokenizer/train_tokenizer.py \
        --data_root_dir /path/to/LEROBOT_LIBERO_DATA --data_mix libero_all_effect \
        --output_dir /path/to/runs/effect_tok --max_steps 40000 --batch_size 64
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from examples.ActionEffect.effect_tokenizer.dataset import (
    build_cached_effect_feature_dataset,
    build_effect_dataset,
    effect_collate,
    effect_feature_collate,
)
from examples.ActionEffect.effect_tokenizer.model import EffectAwareTokenizer, EffectTokenizerConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root_dir", required=True)
    p.add_argument("--data_mix", default="libero_all_effect")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--video_backend", default="torchvision_av")
    p.add_argument("--feature_cache_dir", default=None,
                   help="Optional Qwen-ViT feature cache dir produced by cache_qwen_vit_features.py.")
    p.add_argument("--feature_shard_cache_size", type=int, default=2)
    # optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--target_epochs", type=float, default=0.0,
                   help="If set, override max_steps using dataset_size * target_epochs / global_batch_size.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--num_effect_tokens", type=int, default=4)
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--num_quantizers", type=int, default=1, help="RVQ levels (1 = plain VQ)")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--visual_backend", default="qwen3vl", choices=["dino", "auto", "qwen3vl"])
    p.add_argument("--visual_model_id", default="/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--text_model_id", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--lambda_action", type=float, default=1.0)
    p.add_argument("--lambda_visual", type=float, default=1.0)
    p.add_argument("--lambda_lang", type=float, default=0.0)
    p.add_argument("--lambda_commit", type=float, default=1.0)
    p.add_argument("--lambda_inverse", type=float, default=0.0)
    p.add_argument("--lambda_align", type=float, default=0.0)
    p.add_argument("--vq_beta", type=float, default=0.1)
    p.add_argument("--vq_ema_decay", type=float, default=0.99)
    p.add_argument("--vq_dead_code_threshold", type=float, default=1.0)
    p.add_argument("--no_vq_ema", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = {"video_backend": args.video_backend, "lerobot_version": "v2.0", "image_size": args.image_size}
    if args.feature_cache_dir:
        dataset = build_cached_effect_feature_dataset(
            args.feature_cache_dir,
            args.data_root_dir,
            args.data_mix,
            data_cfg,
            shard_cache_size=args.feature_shard_cache_size,
        )
        collate_fn = effect_feature_collate
        visual_feat_dim = dataset.feature_dim
    else:
        dataset = build_effect_dataset(args.data_root_dir, args.data_mix, data_cfg)
        collate_fn = effect_collate
        visual_feat_dim = 0
    dataset_size = len(dataset)
    global_batch_size = args.batch_size * accelerator.num_processes
    if args.target_epochs and args.target_epochs > 0:
        args.max_steps = max(1, int((dataset_size * args.target_epochs + global_batch_size - 1) // global_batch_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    cfg = EffectTokenizerConfig(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        num_effect_tokens=args.num_effect_tokens,
        num_codes=args.num_codes,
        num_quantizers=args.num_quantizers,
        d_model=args.d_model,
        visual_backend=args.visual_backend,
        visual_model_id=args.visual_model_id,
        visual_feat_dim=visual_feat_dim,
        image_size=args.image_size,
        text_model_id=args.text_model_id,
        lambda_action=args.lambda_action,
        lambda_visual=args.lambda_visual,
        lambda_lang=args.lambda_lang,
        lambda_commit=args.lambda_commit,
        lambda_inverse=args.lambda_inverse,
        lambda_align=args.lambda_align,
        vq_beta=args.vq_beta,
        vq_ema=not args.no_vq_ema,
        vq_ema_decay=args.vq_ema_decay,
        vq_dead_code_threshold=args.vq_dead_code_threshold,
    )
    model = EffectAwareTokenizer(cfg)

    # Only the *trainable* params (frozen backbones excluded automatically).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(scheduler_step: int):
        if scheduler_step < args.warmup_steps:
            return scheduler_step / max(1, args.warmup_steps)
        progress = (scheduler_step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Do not pass the scheduler through accelerator.prepare(). Accelerate wraps
    # schedulers by default and may step/scale them with dataloader processes,
    # which makes this explicit global-step schedule decay too early.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        with open(out / "effect_tokenizer_config.json", "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        run_info = {
            "data_root_dir": args.data_root_dir,
            "data_mix": args.data_mix,
            "feature_cache_dir": args.feature_cache_dir,
            "visual_feat_dim": visual_feat_dim if visual_feat_dim > 0 else None,
            "dataset_size": dataset_size,
            "per_device_batch_size": args.batch_size,
            "num_processes": accelerator.num_processes,
            "global_batch_size": global_batch_size,
            "max_steps": args.max_steps,
            "target_epochs": args.target_epochs if args.target_epochs > 0 else None,
            "estimated_epochs": args.max_steps * global_batch_size / max(1, dataset_size),
            "warmup_steps": args.warmup_steps,
            "base_lr": args.lr,
            "scheduler": "manual_global_step_warmup_cosine",
            "log_every": args.log_every,
            "save_every": args.save_every,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out / "stage1_run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"[Stage1] trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
        print(f"[Stage1] run info: {json.dumps(run_info, ensure_ascii=False)}")

    step = 0
    model.train()
    done = False
    start_time = time.time()
    while not done:
        for batch in loader:
            actions = batch["action"].to(accelerator.device)
            if "f_t" in batch:
                f_t = batch["f_t"].to(accelerator.device)
                f_tpH = batch["f_tpH"].to(accelerator.device)
                out_dict = model(f_t, f_tpH, actions, batch["lang"])
            else:
                obs = batch["obs"].to(accelerator.device)
                future_obs = batch["future_obs"].to(accelerator.device)
                out_dict = model(obs, future_obs, actions, batch["lang"])
            loss = out_dict["loss"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            scheduler_step = int(scheduler.last_epoch)

            if step % args.log_every == 0 and accelerator.is_main_process:
                active_codes = _active_code_string(out_dict["codes"], cfg.num_quantizers)
                completed_steps = step + 1
                elapsed_seconds = time.time() - start_time
                seconds_per_step = elapsed_seconds / max(1, completed_steps)
                remaining_steps = max(0, args.max_steps - completed_steps)
                eta_seconds = seconds_per_step * remaining_steps
                metrics = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "l_action": float(out_dict["l_action"].item()),
                    "l_visual": float(out_dict["l_visual"].item()),
                    "l_inverse": float(out_dict["l_inverse"].item()),
                    "l_lang": float(out_dict["l_lang"].item()),
                    "l_align": float(out_dict["l_align"].item()),
                    "vq_loss": float(out_dict["vq_loss"].item()),
                    "perplexity": float(out_dict["perplexity"].item()),
                    "active_codes": active_codes,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "scheduler_step": scheduler_step,
                    "estimated_epoch": step * global_batch_size / max(1, dataset_size),
                    "elapsed_seconds": elapsed_seconds,
                    "seconds_per_step": seconds_per_step,
                    "eta_seconds": eta_seconds,
                    "eta": _format_duration(eta_seconds),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(out / "stage1_metrics.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(
                    f"step {step:>7} | loss {loss.item():.4f} | act {out_dict['l_action'].item():.4f} "
                    f"| vis {out_dict['l_visual'].item():.4f} | inv {out_dict['l_inverse'].item():.4f} "
                    f"| lang {out_dict['l_lang'].item():.4f} | align {out_dict['l_align'].item():.4f} "
                    f"| vq {out_dict['vq_loss'].item():.4f} | ppl {out_dict['perplexity'].item():.1f} "
                    f"| active {active_codes} | lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| {seconds_per_step:.2f}s/step | elapsed {_format_duration(elapsed_seconds)} "
                    f"| eta {_format_duration(eta_seconds)}"
                )

            if step > 0 and step % args.save_every == 0 and accelerator.is_main_process:
                _save(accelerator, model, ckpt_dir / f"effect_tokenizer_{step}.pt", cfg)

            step += 1
            if step >= args.max_steps:
                done = True
                break

    if accelerator.is_main_process:
        _save(accelerator, model, ckpt_dir / "effect_tokenizer.pt", cfg)
        print("[Stage1] done.")


def _save(accelerator, model, path: Path, cfg: EffectTokenizerConfig):
    unwrapped = accelerator.unwrap_model(model)
    torch.save({"state_dict": unwrapped.state_dict(), "config": dataclasses.asdict(cfg)}, path)
    print(f"[Stage1] saved {path}")


def _active_code_string(codes: torch.Tensor, num_quantizers: int) -> str:
    """Return per-level active-code counts for the current local batch."""
    if codes.ndim == 2:
        codes = codes.unsqueeze(-1)
    counts = [int(torch.unique(codes[..., level]).numel()) for level in range(num_quantizers)]
    return "/".join(str(c) for c in counts)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
