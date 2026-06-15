# Copyright 2025 starVLA community. Licensed under the MIT License.
"""Step-through debug harness for the ActionEffect implementation.

Runs entirely offline on CPU with random tensors (no LIBERO data, no GPU, no
network — the frozen backbones use their deterministic fallbacks). Set
breakpoints in any section below and step into the real modules
(`vq.py`, `feature_extractor.py`, `model.py`, `flow_head.py`).

Run via the ".vscode/launch.json" config "ActionEffect: debug_check (offline)"
or directly:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python examples/ActionEffect/debug_check.py
"""

from __future__ import annotations

import os

# Force the offline fallbacks so this never tries to reach HuggingFace.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from examples.ActionEffect.effect_tokenizer.vq import VectorQuantizer, ResidualVQ
from examples.ActionEffect.effect_tokenizer.feature_extractor import (
    FrozenTextEncoder,
    build_visual_encoder,
)
from examples.ActionEffect.effect_tokenizer.model import (
    EffectAwareTokenizer,
    EffectTokenizerConfig,
    codes_to_effect_string,
)
from examples.ActionEffect.effect_tokenizer.flow_head import ConditionalFlowHead


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n# {title}\n" + "=" * 70)


def main() -> None:
    torch.manual_seed(0)
    B, V, H, Da = 3, 2, 8, 7
    N, C, L, d = 4, 512, 2, 128  # effect queries, codebook size, RVQ levels, d_model

    # --- fake batch (shapes mirror the real LIBERO sample) ----------------
    image_size = 256
    obs = torch.rand(B, V, 3, image_size, image_size)
    future_obs = torch.rand(B, V, 3, image_size, image_size)
    actions = torch.rand(B, H, Da) * 2 - 1
    langs = ["pick up the bowl", "open the top drawer", "put the plate in the rack"]

    # === 1. Plain VQ ======================================================
    banner("1. VectorQuantizer (single-level)")
    vq = VectorQuantizer(num_codes=C, code_dim=d)
    z = torch.randn(B, N, d)
    z_q, codes, vq_loss, m = vq(z)
    print(f"z_q {tuple(z_q.shape)} | codes {tuple(codes.shape)} | vq_loss {vq_loss.item():.4f} | ppl {m['perplexity'].item():.2f}")

    # === 2. Residual VQ (multi-level) =====================================
    banner("2. ResidualVQ (wall-oss style, L levels)")
    rvq = ResidualVQ(num_quantizers=L, num_codes=C, code_dim=d)
    rvq.eval()  # eval() so the EMA codebook is NOT updated mid-call (clean invariant)
    z_q, codes, vq_loss, m = rvq(z)
    print(f"z_q {tuple(z_q.shape)} | codes {tuple(codes.shape)} (B,N,L) | vq_loss {vq_loss.item():.4f}")
    recon = rvq.lookup(codes)
    print(f"lookup(codes) {tuple(recon.shape)} | sum-of-levels == z_q: {torch.allclose(recon, z_q, atol=1e-4)} "
          f"(in train() EMA updates the codebook mid-call, so this is only exact in eval())")

    # === 3. Frozen feature extractors (fallback paths) ====================
    banner("3. Frozen visual + text encoders (offline fallback)")
    vis = build_visual_encoder("dino", "facebook/dinov2-small")
    txt = FrozenTextEncoder("sentence-transformers/all-MiniLM-L6-v2")
    fv = vis(obs.reshape(B * V, 3, image_size, image_size))
    ft = txt(langs, torch.device("cpu"))
    print(f"visual feat {tuple(fv.shape)} (backend={vis._backend}) | text feat {tuple(ft.shape)} (backend={txt._backend})")

    # === 4. EffectAwareTokenizer forward (all losses) =====================
    banner("4. EffectAwareTokenizer.forward")
    cfg = EffectTokenizerConfig(
        action_dim=Da, action_horizon=H, num_effect_tokens=N,
        num_codes=C, num_quantizers=L, d_model=d,
    )
    model = EffectAwareTokenizer(cfg)
    out = model(obs, future_obs, actions, langs)  # <-- step into _fuse / vq / decoders here
    for k in ["loss", "l_action", "l_visual", "l_inverse", "l_lang", "l_align", "vq_loss", "perplexity"]:
        print(f"  {k:12s} = {float(out[k]):.4f}")
    out["loss"].backward()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  backward OK | trainable params = {n_train/1e6:.2f}M | total_effect_vocab = {model.total_effect_vocab}")

    # === 5. Posterior codes + effect-token string =========================
    banner("5. encode_to_codes -> <effect_*> string")
    codes = model.encode_to_codes(obs, future_obs, actions, langs)  # (B, N, L)
    print(f"codes {tuple(codes.shape)}")
    s = codes_to_effect_string(codes[0].tolist(), cfg.num_codes)
    print(f"sample[0] effect string: {s}")
    import re
    ids = [int(x) for x in re.findall(r"<effect_(\d+)>", s)]
    print(f"emitted {len(ids)} tokens (expect N*L={N*L}) | max flat id {max(ids)} < vocab {model.total_effect_vocab}")

    # === 6. XR-1/UniT-style direct action decode ==========================
    banner("6. decode_actions (codes + o_t + l -> actions)")
    dec = model.decode_actions(codes, obs, langs)
    print(f"decoded actions {tuple(dec.shape)}  (B,H,Da)")

    # === 7. Continuous flow-matching head =================================
    banner("7. ConditionalFlowHead (execution_mode=flow)")
    fh = ConditionalFlowHead(action_dim=Da, action_horizon=H, ctx_dim=2048, d_model=d, n_layers=2)
    ctx = torch.randn(B, 2048)  # stands in for the VLM hidden state
    fl = fh.loss(actions, ctx)  # <-- step into to inspect the flow path
    fl.backward()
    samp = fh.sample(ctx, num_steps=5)
    print(f"flow loss {fl.item():.4f} | sampled actions {tuple(samp.shape)} | backward OK")

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
