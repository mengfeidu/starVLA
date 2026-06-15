# Copyright 2025 starVLA community. Licensed under the MIT License.
"""LIBERO dataset that also returns the future observation ``o_{t+H}``.

Reuses starVLA's ``LeRobotSingleDataset`` machinery but overrides ``_pack_sample``
so each item contains both the current and the future frame(s) needed to compute
the visual *effect* target ``Δf``. Used by the Stage-1 tokenizer trainer.
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
)


class EffectLeRobotSingleDataset(LeRobotSingleDataset):
    """Single dataset whose samples carry current + future observations."""

    def _pack_sample(self, data: dict) -> dict:
        cur_images, fut_images = [], []
        image_size = int(getattr(self, "effect_image_size", 256))
        for video_key in self.modality_keys["video"]:
            frames = data[video_key]  # (num_delta, H, W, C); delta = [0, H]
            cur = Image.fromarray(frames[0]).resize((image_size, image_size))
            fut = Image.fromarray(frames[-1]).resize((image_size, image_size))
            cur_images.append(cur)
            fut_images.append(fut)

        language = data[self.modality_keys["language"][0]][0]
        action = np.concatenate([data[k] for k in self.modality_keys["action"]], axis=1).astype(np.float32)

        return {
            "obs": cur_images,
            "future_obs": fut_images,
            "action": action,
            "lang": language,
            "robot_tag": self.tag,
        }

    def get_action_lang_sample(self, index: int) -> dict:
        """Return action/lang for one sample without decoding video frames."""
        trajectory_id, base_index = self.all_steps[index]
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        raw_data = {}
        for key in self.modality_keys["action"]:
            raw_data[key] = self.get_state_or_action(trajectory_id, "action", key, base_index)
        for key in self.modality_keys["language"]:
            raw_data[key] = self.get_language(trajectory_id, key, base_index)
        raw_data = self._apply_action_mode(raw_data)
        data = self.transforms(raw_data)
        language = data[self.modality_keys["language"][0]][0]
        action = np.concatenate([data[k] for k in self.modality_keys["action"]], axis=1).astype(np.float32)
        return {"action": action, "lang": language}


def _make_single(data_root_dir: Path, data_name: str, robot_type: str, data_cfg: dict | None = None):
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type, EmbodimentTag.NEW_EMBODIMENT)
    video_backend = (data_cfg or {}).get("video_backend", "torchvision_av")
    dataset = EffectLeRobotSingleDataset(
        dataset_path=data_root_dir / data_name,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        data_cfg=data_cfg or {},
    )
    dataset.effect_image_size = int((data_cfg or {}).get("image_size", 256))
    return dataset


def build_effect_dataset(data_root_dir: str, data_mix: str, data_cfg: dict | None = None) -> ConcatDataset:
    """Build a ConcatDataset over a named mixture (effect robot_type)."""
    root = Path(data_root_dir)
    datasets = []
    for d_name, _weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        datasets.append(_make_single(root, d_name, robot_type, data_cfg))
    return ConcatDataset(datasets)


class CachedEffectFeatureDataset(Dataset):
    """Dataset over cached Qwen-ViT feature shards plus action/lang from parquet.

    Feature shards contain ``sample_index``, ``f_t`` and ``f_tpH``. Actions and
    language are read without touching video columns, so Stage-1 training avoids
    online video decode and Qwen-ViT forward.
    """

    def __init__(
        self,
        feature_cache_dir: str,
        data_root_dir: str,
        data_mix: str,
        data_cfg: dict | None = None,
        shard_cache_size: int = 2,
    ):
        self.feature_cache_dir = Path(feature_cache_dir)
        with open(self.feature_cache_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.shard_paths = [self.feature_cache_dir / name for name in self.manifest["shards"]]
        self.base_dataset = build_effect_dataset(data_root_dir, data_mix, data_cfg)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, dict] = OrderedDict()

        lengths = []
        for path in self.shard_paths:
            shard = torch.load(path, map_location="cpu")
            lengths.append(int(shard["sample_index"].numel()))
            del shard
        self._cumulative_sizes = np.cumsum(lengths).tolist()

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, index: int) -> dict:
        shard_idx = bisect.bisect_right(self._cumulative_sizes, index)
        prev = 0 if shard_idx == 0 else self._cumulative_sizes[shard_idx - 1]
        offset = index - prev
        shard = self._get_shard(shard_idx)
        sample_index = int(shard["sample_index"][offset])
        action_lang = self._get_action_lang(sample_index)
        return {
            "f_t": shard["f_t"][offset].float(),
            "f_tpH": shard["f_tpH"][offset].float(),
            "action": action_lang["action"],
            "lang": action_lang["lang"],
            "sample_index": sample_index,
        }

    def _get_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._shard_cache:
            shard = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = shard
            return shard
        shard = torch.load(self.shard_paths[shard_idx], map_location="cpu")
        self._shard_cache[shard_idx] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def _get_action_lang(self, sample_index: int) -> dict:
        dataset_idx = bisect.bisect_right(self.base_dataset.cumulative_sizes, sample_index)
        prev = 0 if dataset_idx == 0 else self.base_dataset.cumulative_sizes[dataset_idx - 1]
        local_index = sample_index - prev
        return self.base_dataset.datasets[dataset_idx].get_action_lang_sample(local_index)


def build_cached_effect_feature_dataset(
    feature_cache_dir: str,
    data_root_dir: str,
    data_mix: str,
    data_cfg: dict | None = None,
    shard_cache_size: int = 2,
) -> CachedEffectFeatureDataset:
    return CachedEffectFeatureDataset(feature_cache_dir, data_root_dir, data_mix, data_cfg, shard_cache_size)


def _pil_list_to_tensor(pils: List[Image.Image]) -> torch.Tensor:
    """List[V] of PIL -> (V, 3, H, W) float in [0,1]."""
    arr = np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in pils])  # (V,H,W,3)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def effect_collate(batch: List[dict]) -> dict:
    obs = torch.stack([_pil_list_to_tensor(b["obs"]) for b in batch])           # (B,V,3,H,W)
    future_obs = torch.stack([_pil_list_to_tensor(b["future_obs"]) for b in batch])
    actions = torch.from_numpy(np.stack([b["action"] for b in batch])).float()   # (B,H,Da)
    langs = [b["lang"] for b in batch]
    return {"obs": obs, "future_obs": future_obs, "action": actions, "lang": langs}


def effect_feature_collate(batch: List[dict]) -> dict:
    f_t = torch.stack([b["f_t"] for b in batch]).float()
    f_tpH = torch.stack([b["f_tpH"] for b in batch]).float()
    actions = torch.from_numpy(np.stack([b["action"] for b in batch])).float()
    langs = [b["lang"] for b in batch]
    sample_index = torch.tensor([b["sample_index"] for b in batch], dtype=torch.long)
    return {"f_t": f_t, "f_tpH": f_tpH, "action": actions, "lang": langs, "sample_index": sample_index}
