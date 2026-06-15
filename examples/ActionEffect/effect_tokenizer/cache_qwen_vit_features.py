# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Cache frozen Qwen-ViT features for Stage-1 Effect Tokenizer training.

The online Stage-1 trainer currently runs the frozen visual tower on both
``o_t`` and ``o_{t+H}`` every step. This utility precomputes the same
mean-pooled features used by :class:`EffectAwareTokenizer._encode_views` and
saves them as sharded ``.pt`` files:

    f_t   = mean_view(QwenViT(o_t))
    f_tpH = mean_view(QwenViT(o_{t+H}))

Example:
    accelerate launch examples/ActionEffect/effect_tokenizer/cache_qwen_vit_features.py \
        --data_root_dir /path/to/LEROBOT_LIBERO_DATA --data_mix libero_all_plus_90_effect \
        --output_dir /path/to/qwen_vit_features --batch_size 128
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Sequence

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from examples.ActionEffect.effect_tokenizer.dataset import build_effect_dataset, effect_collate
from examples.ActionEffect.effect_tokenizer.feature_extractor import build_visual_encoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root_dir", required=True)
    p.add_argument("--data_mix", default="libero_all_plus_90_effect")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--video_backend", default="torchvision_av")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--visual_backend", default="qwen3vl", choices=["qwen3vl", "dino", "auto"])
    p.add_argument("--visual_model_id", default="/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--shard_size", type=int, default=8192, help="Number of samples per saved shard on each rank.")
    p.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--save_delta", action="store_true", help="Also save f_tpH - f_t in each shard.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--fail_on_bad_sample", action="store_true",
                   help="Abort on video decode/data errors instead of skipping and logging them.")
    p.add_argument("--max_bad_sample_ratio", type=float, default=0.05,
                   help="Abort if skipped samples exceed this ratio after the warmup check window.")
    p.add_argument("--bad_sample_check_min", type=int, default=512,
                   help="Minimum checked samples before enforcing max_bad_sample_ratio.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


class IndexedSubset(Dataset):
    """Rank-local view of a dataset that keeps original sample indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], skip_bad_samples: bool = True):
        self.dataset = dataset
        self.indices = list(indices)
        self.skip_bad_samples = skip_bad_samples

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample_index = self.indices[idx]
        try:
            sample = dict(self.dataset[sample_index])
            sample["sample_index"] = sample_index
            return sample
        except Exception as exc:
            if not self.skip_bad_samples:
                raise
            return {
                "__bad_sample__": True,
                "sample_index": sample_index,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }


def indexed_effect_collate(batch: list[dict]) -> dict:
    bad_samples = [b for b in batch if b.get("__bad_sample__")]
    good_samples = [b for b in batch if not b.get("__bad_sample__")]
    if good_samples:
        sample_index = torch.tensor([b.pop("sample_index") for b in good_samples], dtype=torch.long)
        out = effect_collate(good_samples)
    else:
        sample_index = torch.empty(0, dtype=torch.long)
        out = {"obs": None, "future_obs": None, "action": None, "lang": []}
    out["sample_index"] = sample_index
    out["bad_samples"] = bad_samples
    return out


@torch.inference_mode()
def encode_views(visual: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Match EffectAwareTokenizer._encode_views: (B,V,3,H,W) -> (B,D)."""
    bsz, num_views = imgs.shape[:2]
    feat = visual(imgs.reshape(bsz * num_views, *imgs.shape[2:]))
    return feat.reshape(bsz, num_views, -1).mean(1)


def main():
    args = parse_args()
    accelerator = Accelerator()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.save_dtype]

    if not args.overwrite and any(out_dir.glob("features_rank*_part*.pt")):
        raise FileExistsError(f"{out_dir} already contains feature shards; pass --overwrite to replace them.")

    if args.overwrite and accelerator.is_main_process:
        for pattern in ["features_rank*_part*.pt", "summary_rank*.json", "bad_samples_rank*.jsonl"]:
            for path in out_dir.glob(pattern):
                path.unlink()
        for path in [out_dir / "manifest.json", out_dir / "run_info.json"]:
            if path.exists():
                path.unlink()
    accelerator.wait_for_everyone()

    data_cfg = {"video_backend": args.video_backend, "lerobot_version": "v2.0", "image_size": args.image_size}
    dataset = build_effect_dataset(args.data_root_dir, args.data_mix, data_cfg)
    dataset_size = len(dataset)
    rank = accelerator.process_index
    world = accelerator.num_processes
    rank_indices = list(range(rank, dataset_size, world))
    rank_dataset = IndexedSubset(dataset, rank_indices, skip_bad_samples=not args.fail_on_bad_sample)
    loader = DataLoader(
        rank_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=indexed_effect_collate,
        pin_memory=True,
    )

    visual = build_visual_encoder(args.visual_backend, args.visual_model_id, args.image_size)
    visual = visual.to(accelerator.device).eval()

    if accelerator.is_main_process:
        run_info = {
            "data_root_dir": args.data_root_dir,
            "data_mix": args.data_mix,
            "dataset_size": dataset_size,
            "num_processes": world,
            "per_device_batch_size": args.batch_size,
            "video_backend": args.video_backend,
            "visual_backend": args.visual_backend,
            "visual_model_id": args.visual_model_id,
            "image_size": args.image_size,
            "save_dtype": args.save_dtype,
            "save_delta": args.save_delta,
            "fail_on_bad_sample": args.fail_on_bad_sample,
            "max_bad_sample_ratio": args.max_bad_sample_ratio,
            "bad_sample_check_min": args.bad_sample_check_min,
            "shard_size": args.shard_size,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"[QwenViTCache] run info: {json.dumps(run_info, ensure_ascii=False)}")

    buffers: dict[str, list[torch.Tensor]] = {"sample_index": [], "f_t": [], "f_tpH": []}
    buffered = 0
    written = 0
    skipped = 0
    part = 0
    start_time = time.time()
    bad_sample_log = out_dir / f"bad_samples_rank{rank:03d}.jsonl"

    def flush(force: bool = False):
        nonlocal buffered, written, part
        if buffered == 0 or (not force and buffered < args.shard_size):
            return
        shard = {
            "sample_index": torch.cat(buffers["sample_index"], dim=0),
            "f_t": torch.cat(buffers["f_t"], dim=0),
            "f_tpH": torch.cat(buffers["f_tpH"], dim=0),
        }
        if args.save_delta:
            shard["delta_f"] = shard["f_tpH"].float() - shard["f_t"].float()
        path = out_dir / f"features_rank{rank:03d}_part{part:05d}.pt"
        torch.save(shard, path)
        written += int(shard["sample_index"].numel())
        part += 1
        buffered = 0
        for values in buffers.values():
            values.clear()

    for step, batch in enumerate(loader):
        bad_samples = batch.get("bad_samples", [])
        if bad_samples:
            skipped += len(bad_samples)
            with open(bad_sample_log, "a") as f:
                for item in bad_samples:
                    f.write(json.dumps(item) + "\n")
            print(f"[rank {rank}] skipped {len(bad_samples)} bad samples; see {bad_sample_log}", flush=True)
        checked = written + buffered + skipped
        if (
            args.max_bad_sample_ratio >= 0
            and checked >= args.bad_sample_check_min
            and skipped / max(1, checked) > args.max_bad_sample_ratio
        ):
            raise RuntimeError(
                f"Bad sample ratio too high on rank {rank}: {skipped}/{checked} "
                f"({skipped / max(1, checked):.2%}) with video_backend={args.video_backend}. "
                f"Check {bad_sample_log}; this usually means the video backend is incompatible "
                "with the dataset rather than isolated corrupt samples."
            )
        if batch["obs"] is None:
            continue

        obs = batch["obs"].to(accelerator.device, non_blocking=True)
        future_obs = batch["future_obs"].to(accelerator.device, non_blocking=True)
        f_t = encode_views(visual, obs).cpu().to(save_dtype)
        f_tpH = encode_views(visual, future_obs).cpu().to(save_dtype)
        sample_index = batch["sample_index"].cpu()

        buffers["sample_index"].append(sample_index)
        buffers["f_t"].append(f_t)
        buffers["f_tpH"].append(f_tpH)
        buffered += int(sample_index.numel())
        flush()

        if step % args.log_every == 0:
            done = written + buffered
            elapsed = time.time() - start_time
            samples_per_sec = done / max(1e-6, elapsed)
            remaining = max(0, len(rank_dataset) - done)
            eta = _format_duration(remaining / max(1e-6, samples_per_sec))
            print(
                f"[rank {rank}] step {step:>6} | cached {done}/{len(rank_dataset)} "
                f"| skipped {skipped} | {samples_per_sec:.1f} samples/s | eta {eta}",
                flush=True,
            )

    flush(force=True)
    rank_summary = {
        "rank": rank,
        "num_samples": written,
        "num_skipped": skipped,
        "num_shards": part,
        "first_index": rank_indices[0] if rank_indices else None,
        "last_index": rank_indices[-1] if rank_indices else None,
    }
    with open(out_dir / f"summary_rank{rank:03d}.json", "w") as f:
        json.dump(rank_summary, f, indent=2)
    print(f"[rank {rank}] done: {json.dumps(rank_summary)}", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shard_files = sorted(str(p.name) for p in out_dir.glob("features_rank*_part*.pt"))
        rank_summaries = []
        for path in sorted(out_dir.glob("summary_rank*.json")):
            with open(path) as f:
                rank_summaries.append(json.load(f))
        manifest = {
            "dataset_size": dataset_size,
            "feature_dim": int(getattr(visual, "feat_dim")),
            "feature_keys": ["sample_index", "f_t", "f_tpH"] + (["delta_f"] if args.save_delta else []),
            "video_backend": args.video_backend,
            "num_shards": len(shard_files),
            "shards": shard_files,
            "rank_summaries": rank_summaries,
            "num_skipped": sum(int(item.get("num_skipped", 0)) for item in rank_summaries),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[QwenViTCache] saved manifest with {len(shard_files)} shards to {out_dir / 'manifest.json'}")


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
