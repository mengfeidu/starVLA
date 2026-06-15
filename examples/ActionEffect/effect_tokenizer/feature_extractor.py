# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Frozen visual + language feature extractors for the effect tokenizer.

The "effect" of an action is measured in a *frozen* feature space ``f`` rather
than at the pixel level (see README §2.3). We default to small HF models
(DINOv2 for vision, MiniLM/DistilBERT for text) and fall back to a deterministic
random projection when the weights cannot be downloaded, so this module always
imports and runs a smoke test offline. Swap in a real backbone for meaningful
features.

All parameters are frozen (``requires_grad_(False)`` + ``eval()``).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class FrozenVisualEncoder(nn.Module):
    """Frozen image encoder returning a pooled feature vector per image.

    Output: ``(B, feat_dim)`` L2-normalized features.
    """

    def __init__(self, model_id: str = "facebook/dinov2-small", image_size: int = 256):
        super().__init__()
        self.model_id = model_id
        self.image_size = image_size
        self._backend = "none"
        self.feat_dim = 384  # dinov2-small default; overwritten on successful load

        try:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = _freeze(AutoModel.from_pretrained(model_id))
            self.feat_dim = int(self.model.config.hidden_size)
            self._backend = "hf"
        except Exception as e:  # offline / missing weights → deterministic fallback
            print(f"[FrozenVisualEncoder] HF load failed ({e!r}); using random-projection fallback.")
            # A fixed (frozen) random conv → pooled projection. Deterministic via manual_seed.
            g = torch.Generator().manual_seed(0)
            self.fallback = _freeze(nn.Conv2d(3, self.feat_dim, kernel_size=16, stride=16))
            with torch.no_grad():
                for p in self.fallback.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            self._backend = "fallback"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: ``(B, 3, H, W)`` float in [0, 1]. Returns ``(B, feat_dim)``."""
        if self._backend == "hf":
            # HF processors expect pixel preprocessing; we approximate with the
            # model's own normalization to stay tensor-friendly and fast.
            mean = torch.tensor(getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]), device=images.device)
            std = torch.tensor(getattr(self.processor, "image_std", [0.229, 0.224, 0.225]), device=images.device)
            x = (images - mean[None, :, None, None]) / std[None, :, None, None]
            out = self.model(pixel_values=x)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:  # CLS token
                feat = out.last_hidden_state[:, 0]
        else:
            feat = self.fallback(images).flatten(2).mean(-1)  # (B, feat_dim)
        return F.normalize(feat, dim=-1)


class FrozenVLMVisionEncoder(nn.Module):
    """Frozen Qwen-VL vision tower — for *VLM-native* visual-action alignment.

    wall-oss aligns action latents to the backbone's own visual features so the
    discrete tokens become a semantic training interface for that backbone
    (paper §2.1.2 / §5.3). Using the Qwen3-VL vision tower here makes the effect
    codes live in the same feature space the downstream QwenEffect backbone uses.

    This path is heavier and version-sensitive (Qwen packs pixels with grid_thw),
    so it is best-effort: any failure falls back to a deterministic projection,
    exactly like :class:`FrozenVisualEncoder`.
    """

    def __init__(self, model_id: str, image_size: int = 256):
        super().__init__()
        self.model_id = model_id
        self.image_size = image_size
        self._backend = "none"
        self.feat_dim = 1152

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id)
            full = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
            self.visual = _freeze(full.model.visual)
            vision_cfg = getattr(full.config, "vision_config", full.config)
            self.feat_dim = int(getattr(vision_cfg, "out_hidden_size", getattr(vision_cfg, "hidden_size", self.feat_dim)))
            self._backend = "qwen"
        except Exception as e:
            print(f"[FrozenVLMVisionEncoder] Qwen vision load failed ({e!r}); random-projection fallback.")
            g = torch.Generator().manual_seed(2)
            self.fallback = _freeze(nn.Conv2d(3, self.feat_dim, kernel_size=16, stride=16))
            with torch.no_grad():
                for p in self.fallback.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            self._backend = "fallback"

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W) in [0,1] -> (B, feat_dim) L2-normalized."""
        if self._backend == "qwen":
            from PIL import Image
            import numpy as np

            pil = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in images]
            inputs = self.processor.image_processor(images=pil, return_tensors="pt")
            pv = inputs["pixel_values"].to(images.device, dtype=next(self.visual.parameters()).dtype)
            grid = inputs.get("image_grid_thw", None)
            grid = grid.to(images.device) if grid is not None else None
            embeds = self.visual(pv, grid_thw=grid)
            if isinstance(embeds, tuple):
                embeds = embeds[0]
            elif hasattr(embeds, "last_hidden_state"):
                embeds = embeds.last_hidden_state
            elif hasattr(embeds, "hidden_states"):
                embeds = embeds.hidden_states[-1]

            # Mean-pool to one feature per image. Qwen-VL versions differ:
            # some return (B, T, D), others return flattened (sum_T, D).
            if embeds.ndim == 3:
                feat = embeds.mean(1)
            elif embeds.ndim == 2 and embeds.shape[0] == images.size(0):
                feat = embeds
            elif grid is not None:
                counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
                feats, off = [], 0
                for n in counts:
                    feats.append(embeds[off:off + n].mean(0))
                    off += n
                feat = torch.stack(feats)
            else:
                feat = embeds.reshape(images.size(0), -1, embeds.shape[-1]).mean(1)
        else:
            feat = self.fallback(images).flatten(2).mean(-1)
        feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(feat, dim=-1)


def build_visual_encoder(backend: str, model_id: str, image_size: int = 256) -> nn.Module:
    """Factory: ``backend`` in {"dino"/"auto", "qwen3vl"}.

    - "dino"/"auto": :class:`FrozenVisualEncoder` over any AutoModel checkpoint
      (DINOv2, SigLIP, ...). SigLIP is a good "VLM-native-ish" choice.
    - "qwen3vl": :class:`FrozenVLMVisionEncoder` using the Qwen-VL vision tower.
    """
    if backend == "qwen3vl":
        return FrozenVLMVisionEncoder(model_id, image_size)
    return FrozenVisualEncoder(model_id, image_size)


class FrozenTextEncoder(nn.Module):
    """Frozen sentence encoder returning a pooled feature per instruction."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_id = model_id
        self._backend = "none"
        self.feat_dim = 384

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = _freeze(AutoModel.from_pretrained(model_id))
            self.feat_dim = int(self.model.config.hidden_size)
            self._backend = "hf"
        except Exception as e:
            print(f"[FrozenTextEncoder] HF load failed ({e!r}); using hashing-bag fallback.")
            self._backend = "fallback"
            self._proj = _freeze(nn.Linear(1024, self.feat_dim, bias=False))
            with torch.no_grad():
                g = torch.Generator().manual_seed(1)
                self._proj.weight.copy_(torch.randn(self._proj.weight.shape, generator=g) * 0.02)

    @torch.no_grad()
    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self._backend == "hf":
            enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            feat = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # mean-pool
        else:
            # Deterministic hashing bag-of-words → fixed projection.
            vecs = []
            for t in texts:
                bag = torch.zeros(1024)
                for tok in str(t).lower().split():
                    bag[hash(tok) % 1024] += 1.0
                vecs.append(bag)
            feat = self._proj(torch.stack(vecs).to(device))
        return F.normalize(feat, dim=-1)
