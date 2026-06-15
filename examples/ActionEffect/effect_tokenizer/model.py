# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Action-first Effect Tokenizer (Stage 1).

The first version deliberately keeps the discrete code posterior action-only:
``a_{t:t+H} -> RVQ codes``. Frozen visual features are used only as a conditional
auxiliary target during training, so the codes learn what the model's own action
chunk tends to do in the current scene without copying future vision into the
token posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import FrozenTextEncoder, build_visual_encoder
from .vq import ResidualVQ


@dataclass
class EffectTokenizerConfig:
    action_dim: int = 7
    action_horizon: int = 8           # H
    num_effect_tokens: int = 4        # N codes per chunk
    num_codes: int = 512              # codebook size C (per RVQ level)
    num_quantizers: int = 1           # L — RVQ levels (1 == plain VQ)
    d_model: int = 256                # = code_dim
    n_heads: int = 8
    n_fusion_layers: int = 2
    n_action_layers: int = 2
    # frozen backbones
    visual_backend: str = "qwen3vl"   # "qwen3vl" (VLM-native) | "dino"/"auto" (AutoModel)
    visual_model_id: str = "/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action"
    visual_feat_dim: int = 0          # >0 means features are precomputed; skip loading visual backbone
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_size: int = 256
    # loss weights
    lambda_action: float = 1.0
    lambda_visual: float = 1.0
    lambda_lang: float = 0.0
    lambda_commit: float = 1.0        # scales the VQ (consistency) term
    lambda_inverse: float = 0.0
    lambda_align: float = 0.0
    vq_beta: float = 0.25
    vq_ema: bool = True
    vq_ema_decay: float = 0.99
    vq_dead_code_threshold: float = 1.0


def _mlp(sizes: List[int], act=nn.GELU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class EffectAwareTokenizer(nn.Module):
    def __init__(self, cfg: Optional[EffectTokenizerConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or EffectTokenizerConfig(**kwargs)
        c = self.cfg
        d = c.d_model

        # --- frozen feature extractors -----------------------------------
        self.visual = None if c.visual_feat_dim > 0 else build_visual_encoder(c.visual_backend, c.visual_model_id, c.image_size)
        self.text = FrozenTextEncoder(c.text_model_id) if c.lambda_lang > 0 else None
        vdim = c.visual_feat_dim if c.visual_feat_dim > 0 else self.visual.feat_dim
        tdim = self.text.feat_dim if self.text is not None else d

        # --- input projections -------------------------------------------
        self.f_proj = _mlp([vdim, d, d])          # f(o_t)
        self.df_proj = _mlp([vdim, d, d])         # Δf = f(o_{t+H}) - f(o_t)
        self.l_proj = _mlp([tdim, d, d]) if self.text is not None else None
        self.act_in = nn.Linear(c.action_dim, d)
        self.act_pos = nn.Parameter(torch.randn(1, c.action_horizon, d) * 0.02)

        # --- action encoder (per-step tokens) ----------------------------
        enc_layer = nn.TransformerEncoderLayer(d, c.n_heads, dim_feedforward=4 * d, batch_first=True)
        self.action_encoder = nn.TransformerEncoder(enc_layer, c.n_action_layers)

        # --- action-only tokenizer encoder (cross-attention into N queries)
        self.effect_queries = nn.Parameter(torch.randn(1, c.num_effect_tokens, d) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d, c.n_heads, dim_feedforward=4 * d, batch_first=True)
        self.fusion = nn.TransformerDecoder(dec_layer, c.n_fusion_layers)

        # --- (Residual) VQ bottleneck ------------------------------------
        self.vq = ResidualVQ(
            c.num_quantizers,
            c.num_codes,
            d,
            beta=c.vq_beta,
            use_ema=c.vq_ema,
            ema_decay=c.vq_ema_decay,
            dead_code_threshold=c.vq_dead_code_threshold,
        )

        # --- decoders -----------------------------------------------------
        # D_a: codes -> action chunk. No visual/language shortcut in the first version.
        self.action_decoder = _mlp([d * c.num_effect_tokens, 4 * d, c.action_horizon * c.action_dim])
        # D_v: pooled codes + f(o_t) -> Δf. Vision is a condition for understanding effect.
        self.visual_decoder = _mlp([2 * d, 4 * d, vdim])
        # Kept for checkpoint/config compatibility; disabled by default.
        self.inverse_decoder = _mlp([2 * d, 4 * d, c.action_horizon * c.action_dim])
        # code -> text space for the InfoNCE language term
        self.code_to_text = _mlp([d * c.num_effect_tokens, d, tdim])
        # latent -> visual space for the wall-oss visual-action alignment term
        self.latent_to_visual = _mlp([d * c.num_effect_tokens, d, vdim])

        # Keep DDP happy: disabled auxiliary branches should not appear as
        # trainable parameters, otherwise they are unused in the loss.
        if c.lambda_inverse <= 0:
            self._freeze_module(self.inverse_decoder)
            self._freeze_module(self.df_proj)
        if c.lambda_lang <= 0:
            self._freeze_module(self.code_to_text)
        if c.lambda_align <= 0:
            self._freeze_module(self.latent_to_visual)
        if c.lambda_visual <= 0:
            self._freeze_module(self.visual_decoder)
            self._freeze_module(self.f_proj)

    @property
    def total_effect_vocab(self) -> int:
        """Number of <effect_*> tokens = levels * codebook size (id = level*C + code)."""
        return self.cfg.num_quantizers * self.cfg.num_codes

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------------
    # feature helpers
    # ---------------------------------------------------------------------
    def _encode_views(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, V, 3, H, W) -> mean-pooled frozen feature (B, vdim)."""
        if self.visual is None:
            raise RuntimeError("This tokenizer was initialized with precomputed visual features.")
        B, V = imgs.shape[:2]
        feat = self.visual(imgs.reshape(B * V, *imgs.shape[2:]))  # (B*V, vdim)
        return feat.reshape(B, V, -1).mean(1)

    def _fuse(self, actions):
        """Run the action-only tokenizer encoder, return latent queries z_e (B, N, d)."""
        a = self.act_in(actions) + self.act_pos                  # (B, H, d)
        a = self.action_encoder(a)                                # (B, H, d)
        q = self.effect_queries.expand(actions.size(0), -1, -1)   # (B, N, d)
        z_e = self.fusion(q, a)                                    # (B, N, d)
        return z_e

    # ---------------------------------------------------------------------
    # training forward
    # ---------------------------------------------------------------------
    def forward(self, obs, future_obs, actions, langs) -> dict:
        """
        obs/future_obs: (B, V, 3, H, W) float in [0,1]
        actions:        (B, H, action_dim) normalized
        langs:          optional list[str] length B; only used when lambda_lang > 0
        """
        if obs.ndim == 2 and future_obs.ndim == 2:
            return self.forward_from_features(obs, future_obs, actions, langs)
        c = self.cfg
        with torch.no_grad():
            f_t = self._encode_views(obs)            # (B, vdim)
            f_fut = self._encode_views(future_obs)   # (B, vdim)
        return self.forward_from_features(f_t, f_fut, actions, langs)

    def forward_from_features(self, f_t, f_fut, actions, langs) -> dict:
        """Training forward with precomputed visual features.

        f_t/f_fut: (B, vdim), matching the output of ``_encode_views``.
        """
        c = self.cfg
        device = actions.device
        f_t = f_t.to(device=device, dtype=actions.dtype).float()
        f_fut = f_fut.to(device=device, dtype=actions.dtype).float()
        df = f_fut - f_t
        with torch.no_grad():
            g_l = self.text(langs, device) if self.text is not None else None

        z_e = self._fuse(actions)                    # (B, N, d)
        z_e_flat = z_e.reshape(z_e.size(0), -1)      # (B, N*d) — pre-VQ latent for alignment
        z_q, codes, vq_loss, vq_metrics = self.vq(z_e)
        z_flat = z_q.reshape(z_q.size(0), -1)        # (B, N*d)
        z_pooled = z_q.mean(dim=1)                    # (B, d)

        a_hat = self.action_decoder(z_flat).reshape(actions.shape)
        df_hat = self.visual_decoder(torch.cat([z_pooled, self.f_proj(f_t)], dim=-1))
        if c.lambda_inverse > 0:
            a_inv = self.inverse_decoder(torch.cat([self.f_proj(f_t), self.df_proj(df)], dim=-1)).reshape(actions.shape)
        latent_vis = F.normalize(self.latent_to_visual(z_e_flat), dim=-1) if c.lambda_align > 0 else None

        # --- losses ------------------------------------------------------
        l_action = F.smooth_l1_loss(a_hat, actions)
        l_visual = (1 - F.cosine_similarity(df_hat, df, dim=-1).mean()) + F.mse_loss(df_hat, df)
        l_inverse = F.smooth_l1_loss(a_inv, actions) if c.lambda_inverse > 0 else actions.new_zeros(())
        if self.text is not None and c.lambda_lang > 0:
            code_txt = F.normalize(self.code_to_text(z_flat), dim=-1)
            l_lang = self._info_nce(code_txt, g_l)
        else:
            l_lang = actions.new_zeros(())
        # wall-oss visual-action alignment: pull the action/effect latent toward the
        # (frozen) visual feature of the current observation, in-batch InfoNCE.
        l_align = self._info_nce(latent_vis, f_t) if c.lambda_align > 0 else actions.new_zeros(())

        total = (
            c.lambda_action * l_action
            + c.lambda_visual * l_visual
            + c.lambda_lang * l_lang
            + c.lambda_commit * vq_loss
            + c.lambda_inverse * l_inverse
            + c.lambda_align * l_align
        )
        return {
            "loss": total,
            "l_action": l_action.detach(),
            "l_visual": l_visual.detach(),
            "l_lang": l_lang.detach(),
            "l_inverse": l_inverse.detach(),
            "l_align": l_align.detach(),
            "vq_loss": vq_loss.detach(),
            "perplexity": vq_metrics["perplexity"],
            "codes": codes.detach(),
        }

    @staticmethod
    def _info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        """Symmetric InfoNCE between two L2-normalized batches (in-batch negatives)."""
        if a.size(0) < 2:
            return a.new_zeros(())
        logits = a @ b.t() / temp
        target = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    # ---------------------------------------------------------------------
    # inference: posterior effect codes (Stage-2 target generation)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def encode_to_codes(self, obs, future_obs, actions, langs) -> torch.Tensor:
        """Return action-derived effect tokens, shape (B, N, L) long (L = RVQ levels).

        ``obs``/``future_obs``/``langs`` are accepted for QwenEffect compatibility but
        intentionally ignored by the first action-first tokenizer.
        """
        self.eval()
        z_e = self._fuse(actions)
        _, codes, _, _ = self.vq(z_e)
        return codes

    @torch.no_grad()
    def decode_actions(self, codes: torch.Tensor, obs, langs) -> torch.Tensor:
        """Decode normalized actions directly from action-first effect codes.

        codes: (B, N, L) long. Returns normalized actions (B, H, action_dim).
        """
        self.eval()
        z_q = self.vq.lookup(codes)                       # (B, N, d)
        z_flat = z_q.reshape(z_q.size(0), -1)
        return self.action_decoder(z_flat).reshape(-1, self.cfg.action_horizon, self.cfg.action_dim)


def codes_to_effect_string(codes, num_codes: int) -> str:
    """Map RVQ effect codes to the VLM special-token string.

    ``codes`` is shaped ``(N, L)`` (N effect queries, L RVQ levels). Each
    (level ``l``, code ``c``) maps to a single flat vocab id ``l*num_codes + c``,
    emitted level-by-level within each query so the sequence is unambiguous to
    regroup. Returns e.g. ``<effect_12><effect_531>...``.
    """
    parts = []
    for query in codes:                       # length N
        for level, c in enumerate(query):     # length L
            parts.append(f"<effect_{level * num_codes + int(c)}>")
    return "".join(parts)
