# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Compact conditional flow-matching action head.

Optional *continuous executor* for QwenEffect (``execution_mode="flow"``),
mirroring wall-oss: the discrete effect tokens act as a latent plan that shapes
the backbone, while precise actions are produced by a continuous flow-matching
pathway conditioned on the VLM hidden state (and, optionally, the effect codes).

Self-contained (no diffusers dependency). Linear Gaussian path
``x_t = t·x + (1-t)·ε``; the network regresses the velocity ``x - ε`` and, with
``action_space_supervision``, also the clean action ``x`` (wall-oss §2.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0,1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1)))
    args = t[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalFlowHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 8,
        ctx_dim: int = 2048,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        action_space_supervision: bool = True,
        lambda_x: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_space_supervision = action_space_supervision
        self.lambda_x = lambda_x

        self.x_in = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, action_horizon, d_model) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.net = nn.TransformerEncoder(layer, n_layers)
        self.v_out = nn.Linear(d_model, action_dim)   # velocity
        self.x_out = nn.Linear(d_model, action_dim)    # optional clean-action prediction
        self.d_model = d_model

    def _backbone(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.x_in(x_t) + self.pos                                   # (B,H,d)
        cond = self.t_embed(_sinusoidal_time_embed(t, self.d_model)) + self.ctx_proj(ctx)  # (B,d)
        h = h + cond[:, None, :]
        return self.net(h)                                             # (B,H,d)

    def loss(self, actions: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Flow-matching loss for clean actions (B,H,Da) given context (B,ctx_dim)."""
        B = actions.size(0)
        eps = torch.randn_like(actions)
        t = torch.rand(B, device=actions.device)
        x_t = t[:, None, None] * actions + (1 - t[:, None, None]) * eps
        h = self._backbone(x_t, t, ctx)
        v_pred = self.v_out(h)
        l_v = F.mse_loss(v_pred, actions - eps)
        if self.action_space_supervision:
            x_pred = self.x_out(h)
            return l_v + self.lambda_x * F.smooth_l1_loss(x_pred, actions)
        return l_v

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Euler-integrate the velocity field. Returns (B,H,Da)."""
        B = ctx.size(0)
        x = torch.randn(B, self.action_horizon, self.action_dim, device=ctx.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=ctx.device)
            v = self.v_out(self._backbone(x, t, ctx))
            x = x + v * dt
        return x
