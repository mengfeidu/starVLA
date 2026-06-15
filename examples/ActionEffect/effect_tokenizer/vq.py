# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Minimal Vector-Quantization bottleneck.

A dependency-free re-implementation of the standard VQ-VAE codebook with a
straight-through estimator and commitment loss (van den Oord et al., 2017),
plus optional EMA codebook updates. This is the discrete bottleneck that turns
the continuous fused latent ``z_e`` into a small set of *effect tokens*.

We deliberately avoid ``lucidrains/vector-quantize-pytorch`` (used by UniT) to
keep this scaffold free of extra dependencies, while exposing a compatible API.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantizer with straight-through gradients.

    Args:
        num_codes: codebook size ``C`` (number of distinct effect tokens).
        code_dim: dimensionality of each codebook vector.
        beta: commitment loss weight (the ``β`` in the VQ-VAE paper).
        use_ema: if True, update the codebook with exponential moving averages
            instead of the codebook loss term (more stable for small batches).
        ema_decay: EMA decay factor.
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 256,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        codebook = torch.randn(num_codes, code_dim) * (code_dim ** -0.5)
        if use_ema:
            # Codebook is a buffer updated by EMA, not by autograd.
            self.register_buffer("codebook", codebook)
            self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("ema_codebook", codebook.clone())
        else:
            self.codebook = nn.Parameter(codebook)

    def forward(self, z_e: torch.Tensor):
        """Quantize ``z_e``.

        Args:
            z_e: float tensor of shape ``(..., code_dim)`` (any leading dims).

        Returns:
            z_q:      quantized tensor, same shape as ``z_e`` (straight-through).
            codes:    long tensor of code indices, shape == ``z_e.shape[:-1]``.
            vq_loss:  scalar commitment(+codebook) loss.
            metrics:  dict with ``perplexity`` (codebook usage diagnostic).
        """
        input_shape = z_e.shape
        flat = z_e.reshape(-1, self.code_dim)  # (M, code_dim)

        # Squared L2 distance to every codebook entry.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )  # (M, C)
        codes_flat = dist.argmin(dim=1)  # (M,)
        z_q_flat = self.codebook[codes_flat]  # (M, code_dim)

        # Losses.
        commitment = F.mse_loss(z_q_flat.detach(), flat)
        if self.use_ema:
            if self.training:
                self._ema_update(flat, codes_flat)
            vq_loss = self.beta * commitment
        else:
            codebook_loss = F.mse_loss(z_q_flat, flat.detach())
            vq_loss = codebook_loss + self.beta * commitment

        # Straight-through estimator.
        z_q_flat = flat + (z_q_flat - flat).detach()

        z_q = z_q_flat.reshape(input_shape)
        codes = codes_flat.reshape(input_shape[:-1])

        # Perplexity (how many codes are effectively used).
        with torch.no_grad():
            one_hot = F.one_hot(codes_flat, self.num_codes).float()
            probs = one_hot.mean(0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        return z_q, codes, vq_loss, {"perplexity": perplexity.detach()}

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes_flat: torch.Tensor) -> None:
        one_hot = F.one_hot(codes_flat, self.num_codes).type_as(flat)  # (M, C)
        cluster_size = one_hot.sum(0)  # (C,)
        embed_sum = one_hot.t() @ flat  # (C, code_dim)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_codebook.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_codebook / cluster_size.unsqueeze(1))
        self._replace_dead_codes(flat)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.codebook, src=0)
            dist.broadcast(self.ema_codebook, src=0)
            dist.broadcast(self.ema_cluster_size, src=0)

    @torch.no_grad()
    def _replace_dead_codes(self, flat: torch.Tensor) -> None:
        if self.dead_code_threshold <= 0 or flat.numel() == 0:
            return
        dead = self.ema_cluster_size < self.dead_code_threshold
        if not dead.any():
            return
        num_dead = int(dead.sum().item())
        replace = flat[torch.randint(0, flat.size(0), (num_dead,), device=flat.device)].detach()
        noise = torch.randn_like(replace) * 1e-4
        self.codebook[dead] = replace + noise
        self.ema_codebook[dead] = self.codebook[dead]
        self.ema_cluster_size[dead] = self.dead_code_threshold

    @torch.no_grad()
    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Map code indices back to codebook vectors."""
        return self.codebook[codes]


class ResidualVQ(nn.Module):
    """Multi-level Residual Vector Quantization (wall-oss / lee2022autoregressive).

    Stacks ``num_quantizers`` codebooks; each level quantizes the residual left
    by the previous levels, so early levels capture coarse motion structure and
    later levels capture fine residual corrections. Reduces to a plain
    :class:`VectorQuantizer` when ``num_quantizers == 1``.

    The per-token code index now has a *level* dimension: ``codes`` has shape
    ``(..., num_quantizers)``. Each codebook keeps its own ``[0, num_codes)``
    index space; callers that need a single flat vocab id should offset by
    ``level * num_codes`` (see ``model.codes_to_effect_string``).
    """

    def __init__(
        self,
        num_quantizers: int = 2,
        num_codes: int = 512,
        code_dim: int = 256,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.layers = nn.ModuleList(
            VectorQuantizer(
                num_codes,
                code_dim,
                beta=beta,
                use_ema=use_ema,
                ema_decay=ema_decay,
                dead_code_threshold=dead_code_threshold,
            )
            for _ in range(num_quantizers)
        )

    def forward(self, z_e: torch.Tensor):
        """Quantize ``z_e`` (..., code_dim) with residual quantization.

        Returns ``z_q`` (same shape), ``codes`` (..., num_quantizers),
        ``vq_loss`` (summed over levels), and ``metrics`` (mean perplexity).
        """
        residual = z_e
        z_q = torch.zeros_like(z_e)
        codes, vq_loss, ppls = [], z_e.new_zeros(()), []
        for layer in self.layers:
            q, c, loss, m = layer(residual)
            residual = residual - q          # ST-quantized numerically equals codebook vector
            z_q = z_q + q
            codes.append(c)
            vq_loss = vq_loss + loss
            ppls.append(m["perplexity"])
        codes = torch.stack(codes, dim=-1)   # (..., L)
        perplexity = torch.stack(ppls).mean() if ppls else z_e.new_zeros(())
        return z_q, codes, vq_loss, {"perplexity": perplexity}

    @torch.no_grad()
    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (..., num_quantizers) long -> summed codebook vectors (..., code_dim)."""
        out = None
        for l, layer in enumerate(self.layers):
            vec = layer.lookup(codes[..., l])
            out = vec if out is None else out + vec
        return out
