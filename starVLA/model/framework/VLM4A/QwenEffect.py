# Copyright 2025 starVLA community. Licensed under the MIT License.
"""QwenEffect — effect-conditioned autoregressive action prediction (Stage 2).

This is the only file of the ActionEffect recipe that lives inside the
``starVLA`` package: it exists so the framework registry can auto-discover and
register ``QwenEffect``. All heavy logic (the Effect-Aware Tokenizer + the
optional flow head) is lazily imported from ``examples/ActionEffect/`` to keep
this file thin and to avoid loading the frozen backbones unless QwenEffect is
actually used.

Idea (README §5): the assistant target becomes

    <effect_*> ... <effect_*>   <robot_action_*> ... <robot_action_*>
    └── latent "physical intent" plan ──┘ └──── FAST low-level actions ────┘

Ground-truth effect tokens are produced on the fly by the frozen Stage-1 encoder
(which needs the future frame o_{t+H}; provided by the EffectLibero data config +
the ``include_future_obs`` dataloader hook). At inference the VLM *generates* the
effect tokens first, then the action tokens — no future frames required.

Two execution modes (``framework.effect.execution_mode``):
  - "fast"  : effect plan tokens followed by FAST action tokens (pure AR; default).
  - "flow"  : effect plan tokens (CE) + a continuous flow-matching head conditioned
              on the VLM hidden state produces the executed actions (wall-oss style).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenFast import Qwenvl_Fast
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100


def _import_effect_tokenizer():
    """Lazy, path-robust import of the Stage-1 tokenizer from examples/."""
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.ActionEffect.effect_tokenizer.model import (  # noqa: E402
        EffectAwareTokenizer,
        EffectTokenizerConfig,
        codes_to_effect_string,
    )

    return EffectAwareTokenizer, EffectTokenizerConfig, codes_to_effect_string


@FRAMEWORK_REGISTRY.register("QwenEffect")
class Qwenvl_Effect(Qwenvl_Fast):
    """Effect-token (+ FAST-token / + flow head) autoregressive VLA.

    Extra ``framework.effect`` config keys:
        tokenizer_ckpt:    path to Stage-1 ``effect_tokenizer.pt`` (required for training).
        decode_mode:       "ar" (default) or "decoder" (decode actions with frozen D_a).
        execution_mode:    "fast" (default) or "flow".
        lambda_flow:       weight of the flow-matching loss (flow mode).
        flow_sample_steps: Euler steps at inference (flow mode).
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        EffectAwareTokenizer, EffectTokenizerConfig, codes_to_effect_string = _import_effect_tokenizer()
        self._codes_to_effect_string = codes_to_effect_string

        eff_cfg = self.config.framework.get("effect", {}) or {}
        ckpt_path = eff_cfg.get("tokenizer_ckpt", None)
        self.decode_mode = eff_cfg.get("decode_mode", "ar")

        if ckpt_path and Path(ckpt_path).is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            tok_cfg = EffectTokenizerConfig(**ckpt["config"])
            self.effect_tokenizer = EffectAwareTokenizer(tok_cfg)
            self.effect_tokenizer.load_state_dict(ckpt["state_dict"], strict=False)
            logger.info(f"[QwenEffect] loaded effect tokenizer from {ckpt_path}")
        else:
            logger.warning(
                "[QwenEffect] no valid `framework.effect.tokenizer_ckpt`; "
                "initializing a RANDOM effect tokenizer (debug only)."
            )
            self.effect_tokenizer = EffectAwareTokenizer(EffectTokenizerConfig())

        # The Stage-1 tokenizer is frozen during Stage-2 training.
        self.effect_tokenizer.eval()
        for p in self.effect_tokenizer.parameters():
            p.requires_grad_(False)

        # Effect token id range in the VLM vocab (appended after FAST tokens).
        # With RVQ there are (num_quantizers * num_codes) effect tokens.
        self._num_codes = int(self.effect_tokenizer.cfg.num_codes)
        self._n_effect = int(self.effect_tokenizer.cfg.num_effect_tokens)
        self._n_levels = int(self.effect_tokenizer.cfg.num_quantizers)
        total_vocab = self.effect_tokenizer.total_effect_vocab
        tok = self.qwen_vl_interface.processor.tokenizer
        self._EFFECT_TOKEN_MIN = tok.convert_tokens_to_ids("<effect_0>")
        self._EFFECT_TOKEN_MAX = tok.convert_tokens_to_ids(f"<effect_{total_vocab - 1}>")
        if self._EFFECT_TOKEN_MIN is None or self._EFFECT_TOKEN_MIN < 0:
            logger.warning(
                "[QwenEffect] <effect_*> tokens not found in tokenizer. Run "
                "examples/ActionEffect/train_files/add_effect_tokens.py first."
            )

        # Execution mode: "fast" (discrete FAST tokens) or "flow" (continuous head).
        self.execution_mode = eff_cfg.get("execution_mode", "fast")
        self.lambda_flow = float(eff_cfg.get("lambda_flow", 1.0))
        self.flow_sample_steps = int(eff_cfg.get("flow_sample_steps", 10))
        self.flow_head = None
        if self.execution_mode == "flow":
            from examples.ActionEffect.effect_tokenizer.flow_head import ConditionalFlowHead

            ctx_dim = int(self.qwen_vl_interface.model.config.hidden_size)
            self.flow_head = ConditionalFlowHead(
                action_dim=int(self.config.framework.action_model.action_dim),
                action_horizon=self.action_horizon,
                ctx_dim=ctx_dim,
            )
            logger.info(f"[QwenEffect] execution_mode=flow, flow head ctx_dim={ctx_dim}")

    # ------------------------------------------------------------------
    @staticmethod
    def _pils_to_tensor(batch_images: List[List]) -> torch.Tensor:
        """[[PIL view]*V] * B -> (B, V, 3, H, W) float in [0,1]."""
        out = []
        for views in batch_images:
            arr = np.stack([np.asarray(v.convert("RGB"), dtype=np.float32) / 255.0 for v in views])
            out.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
        return torch.stack(out)

    def _compute_effect_codes(self, examples) -> List[List[int]]:
        device = self.qwen_vl_interface.model.device
        obs = self._pils_to_tensor([ex["image"] for ex in examples]).to(device)
        future = self._pils_to_tensor([ex["future_image"] for ex in examples]).to(device)
        actions = torch.from_numpy(np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in examples])).to(device)
        langs = [ex["lang"] for ex in examples]
        self.effect_tokenizer.to(device)
        with torch.no_grad():
            codes = self.effect_tokenizer.encode_to_codes(obs, future, actions, langs)  # (B, N, L)
        return codes.cpu().tolist()

    def _effect_str(self, codes_2d) -> str:
        return self._codes_to_effect_string(codes_2d, self._num_codes)

    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs):
        if examples is None or "future_image" not in examples[0]:
            raise ValueError(
                "QwenEffect.forward needs `future_image` in each sample. Set "
                "datasets.vla_data.include_future_obs=true and use a *_effect data_mix."
            )
        if self.execution_mode == "flow":
            return self._forward_flow(examples)
        return self._forward_fast(examples)

    def _forward_fast(self, examples):
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]

        # Effect plan tokens (ground truth from frozen Stage-1 encoder).
        effect_codes = self._compute_effect_codes(examples)
        effect_strs = [self._effect_str(codes) for codes in effect_codes]

        # FAST low-level action tokens, appended after the effect plan.
        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)
        fast_strs = [self.map_fast_token_to_vlm_action(t) for t in batch_fast_tokens]
        solutions = [e + f for e, f in zip(effect_strs, fast_strs)]

        qwen_inputs = self._build_inputs_with_effect(batch_images, instructions, solutions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=False, return_dict=True
            )
        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)
        return {"action_loss": loss}

    def _forward_flow(self, examples):
        """Effect tokens are the predicted plan (CE); actions come from the flow head."""
        device = self.qwen_vl_interface.model.device
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = torch.from_numpy(
            np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in examples])
        ).to(device)

        effect_codes = self._compute_effect_codes(examples)
        effect_strs = [self._effect_str(codes) for codes in effect_codes]

        # Assistant = effect plan only; supervise effect tokens, read hidden context.
        qwen_inputs = self._build_inputs_with_effect(batch_images, instructions, effect_strs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
        effect_ce = outputs.loss
        if effect_ce is None or torch.isnan(effect_ce):
            effect_ce = torch.tensor(0.0, device=device)

        ctx = outputs.hidden_states[-1][:, -1, :].float()   # (B, vlm_hidden)
        l_flow = self.flow_head.loss(actions, ctx)
        return {"action_loss": effect_ce + self.lambda_flow * l_flow,
                "effect_ce": effect_ce.detach(), "flow_loss": l_flow.detach()}

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        """fast mode -> inherit FAST decoding; flow mode -> plan tokens + flow sampling."""
        if self.execution_mode != "flow":
            return super().predict_action(examples, **kwargs)

        from deployment.model_server.tools.image_tools import to_pil_preserve

        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        max_new = self._n_effect * self._n_levels
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = self.qwen_vl_interface.model.generate(**qwen_inputs, max_new_tokens=max_new, do_sample=False)
            out = self.qwen_vl_interface.model(input_ids=gen, output_hidden_states=True, return_dict=True)
        ctx = out.hidden_states[-1][:, -1, :].float()
        actions = self.flow_head.sample(ctx, num_steps=self.flow_sample_steps)
        return {"normalized_actions": actions.cpu().numpy()}

    def _build_inputs_with_effect(self, images, instructions, solutions):
        """Like QWen3's build_qwenvl_inputs but supervises BOTH effect and action tokens.

        We mask everything before the first *special* token (effect or FAST), so the
        whole latent-plan + action span contributes to the cross-entropy loss.
        """
        interface = self.qwen_vl_interface
        processor = interface.processor
        messages = []
        for imgs, instruction, solution in zip(images, instructions, solutions):
            content = [{"type": "image", "image": img} for img in imgs]
            if "CoT_prompt" in self.config.datasets.vla_data:
                prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
            else:
                prompt = instruction
            content.append({"type": "text", "text": prompt})
            messages.append([
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": solution}]},
            ])

        batch_inputs = processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        lo = min(int(self._EFFECT_TOKEN_MIN), int(interface._ACTION_TOKEN_MIN))
        hi = max(int(self._EFFECT_TOKEN_MAX), int(interface._ACTION_TOKEN_MAX))
        labels = batch_inputs["input_ids"].clone()
        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= lo) & (seq <= hi)
            nz = torch.nonzero(mask_seq, as_tuple=False)
            if nz.numel() > 0:
                seq[: nz[0].item()] = IGNORE_INDEX
            else:
                seq[:] = IGNORE_INDEX
        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch_inputs["labels"] = labels
        return batch_inputs.to(interface.model.device)
