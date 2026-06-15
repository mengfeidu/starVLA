# ActionEffect — Effect-Aware Action Tokenizer for starVLA / QwenFast

This folder is a **self-contained research scaffold** that adds an *action-effect token*
representation on top of starVLA's `QwenFast` model, trained/evaluated on **LIBERO**.

It is a synthesis (and a *corrected, feasible* redesign) of the action-tokenizer ideas in
three recent works:

| Work | Tokenizer idea | What we borrow |
| --- | --- | --- |
| **wall-x / WALL-OSS** ([repo](https://github.com/X-Square-Robot/wall-x)) | `FAST` (DCT+BPE) discrete action tokens + a flow-matching branch, predicted autoregressively by a Qwen2.5-VL MoE. | The action token channel itself — starVLA's `QwenFast` already reproduces this with `physical-intelligence/fast`. We keep it and *condition* it on effect tokens. |
| **XR-1** ([repo](https://github.com/Open-X-Humanoid/XR-1)) | **UVMC** = *Unified Vision-Motion Codes*: a Stage-1 VQ codebook that jointly encodes visual dynamics + motion; Stage-2 VLA predicts the codes; Stage-3 decodes actions (built on Moto/QueST/Pi0). | The 3-stage recipe: *learn discrete vision-motion codes → predict them with the VLM → decode actions.* |
| **UniT** ([repo](https://github.com/xpeng-robotics/UniT)) | **Unified Latent Action Tokenizer via Visual Anchoring**: a *tri-branch cross-reconstruction* VQ-VAE — `action→vision` (anchor kinematics to outcomes), `vision→action` (filter visual confounders), and a `fusion` branch into a shared discrete codebook of "physical intents". | The cross-reconstruction objective and the visual-anchoring philosophy: *heterogeneous kinematics share consistent visual consequences.* This is the core of our "effect" idea. |

The common thread across all three: **discretize *the consequence of acting*, not just the
action vector.** That is exactly what an "action-effect token" should be.

---

## 1. What an *action-effect token* is

A FAST token answers *"what numbers does the arm output?"*. An **effect token** answers
*"what does the world look like after this motion, and what intent does that encode?"*.

Concretely, an effect token is a **discrete code** `z_e ∈ {0..C-1}` (a small number `N` of
them per action chunk) produced by a VQ bottleneck that has *seen the future*
`(o_t, l, a_{t:t+H}, o_{t+1:t+H})` at training time, and is **forced to explain the visual
change** `Δf = f(o_{t+H}) − f(o_t)` in a *frozen* feature space `f` (DINOv2 / SigLIP), the
*action* `a_{t:t+H}`, and the *language* `l`.

Because the code must reconstruct the action **and** predict the visual effect **and** align
with language, it captures a compact, embodiment-agnostic *"physical intent"* — the thing the
VLM should plan in before emitting low-level FAST tokens.

---

## 2. Corrections to the original Stage-1 draft

The draft proposed:

```
z_e          = E_φ(o_t, l, a_{t:t+H}, o_{t+1:t+H})
â_{t:t+H}    = D_a(z_e, o_t, l)
ô_{t+1:t+H}  = D_v(z_e, o_t, l)
L = L_action + λ_v L_visual + λ_l L_lang + λ_c L_consistency
```

It is a good skeleton but **not yet trainable / deployable** as written. Fixes applied here:

1. **`z_e` must be discrete to be a "token".** The draft's `z_e` is continuous. We insert a
   **VQ bottleneck** (`effect_tokenizer/vq.py`): `z_e → nearest codebook entry → code id`.
   Those ids become `<effect_*>` special tokens in the Qwen vocabulary, parallel to
   `<robot_action_*>`. → *Without this you cannot "predict effect tokens" with an LLM.*

2. **Shortcut / information-leakage collapse.** If `z_e` may freely copy `a_{t:t+H}`, then
   `D_a` reconstructs trivially and the visual/effect signal is ignored (the failure UniT's
   `vision→action` branch was designed to prevent). Fixes:
   - **Low-capacity bottleneck**: small `N` (e.g. 4 codes/chunk) and small codebook
     (e.g. `C = 512`).
   - **Asymmetric decoders**: `D_a` and `D_v` *also* receive `(o_t, l)`, so the code only has
     to carry the *residual intent*, not re-encode the observation.
   - **Cross-reconstruction (UniT)**: an auxiliary `vision→action` inverse-dynamics head
     `D_inv(f(o_t), Δf) → a` makes the code action-relevant and filters visual confounders.

3. **`L_visual` should be feature-level, not pixel-level.** The draft already hints at this.
   We make it explicit: **do not reconstruct pixels and then re-encode.** `D_v` directly
   regresses the *frozen-feature delta* `Δf = f(o_{t+H}) − f(o_t)` (cosine + L2). This is
   cheaper, more stable, and is the actual "object-/relation-level visual change" the draft
   wants. `f` is frozen DINOv2/SigLIP (no grad).

4. **`L_lang` made concrete.** The draft's `contrast(z_e, l, Δo)` is under-specified. We use
   **InfoNCE** between the pooled code embedding and a *frozen* text embedding of `l`,
   with in-batch negatives. (LIBERO has limited language diversity, so `λ_l` defaults small
   and `L_lang` can be disabled.)

5. **`L_consistency` defined.** We define it as the **VQ commitment + codebook loss**
   (the term that actually makes the discrete bottleneck train) plus the optional
   inverse-dynamics term from (2). This is what keeps encoder outputs and codebook entries
   consistent.

6. **Deployment gap closed.** At deployment you do **not** have `o_{t+1:t+H}`. The Stage-1
   encoder (which needs the future) is a *training-only posterior*. The **VLM is the prior**
   `p(z_e | o_t, l)` learned in Stage 2 — it predicts the effect tokens autoregressively
   before the action tokens, so no future frames are needed at inference.

Final objective actually optimized in `effect_tokenizer/model.py`:

```
L = L_action                      (Smooth-L1 on â_{t:t+H})
  + λ_v   · L_visual              (cosine + L2 on predicted Δf)
  + λ_l   · L_lang                (InfoNCE: code ↔ frozen text emb)
  + λ_c   · L_vq                  (RVQ commitment / consistency)
  + λ_inv · L_inverse             (vision→action cross-reconstruction, UniT)
  + λ_a   · L_align               (InfoNCE: effect latent ↔ visual feat, wall-oss)
```

---

## 3. Architecture (Stage 1)

```
                       ┌─────────────────────────── frozen f (DINOv2/SigLIP) ──────────────┐
 o_t (multi-view) ─────┤ f(o_t) ───────────────────────────────────────────────┐          │
 o_{t+H}          ─────┤ f(o_{t+H}) ── Δf = f(o_{t+H}) − f(o_t) ──┐             │          │
                       └─────────────────────────────────────────┼─────────────┼──────────┘
 a_{t:t+H} ── ActionEncoder ──────────────────────────┐          │             │
 l ── frozen TextEncoder ── E_l ──────────────┐        │          │             │
                                              ▼        ▼          ▼             │
                                       FusionEncoder E_φ  (cross-attn)          │
                                              │                                 │
                                         VQ bottleneck  → z_e = [code_1..code_N]│  (the effect tokens)
                                              │                                 │
                       ┌──────────────────────┼───────────────────────┐        │
                       ▼                       ▼                        ▼        │
              D_a(z_e, f(o_t), l)     D_v(z_e, f(o_t), l)       D_inv(f(o_t),Δf) │
                 → â_{t:t+H}             → Δf̂                      → â (aux)     │
                 L_action                L_visual                  L_inverse     │
```

Everything trainable is small (≈ a few M params). `f` and the text encoder are frozen.

## 4. Pipeline (3 stages, mirrors XR-1)

```
Stage 1  effect_tokenizer/train_tokenizer.py      learn the VQ effect codebook on LIBERO
Stage 1b effect_tokenizer/export_effect_tokens.py add <effect_*> tokens to Qwen vocab
Stage 2  framework/QwenEffect.py (via train_starvla) VLM predicts <effect_*> then <robot_action_*>
Stage 3  (optional) freeze VLM, fine-tune action decoding only — reuse QwenFast finetune
```

### Run

```bash
# Stage 1 — train the effect tokenizer (standalone, accelerate)
bash examples/ActionEffect/train_files/run_stage1_tokenizer.sh

# Stage 1b — bake <effect_*> tokens into the Action checkpoint
python examples/ActionEffect/train_files/add_effect_tokens.py \
  --model-id  <Qwen3-VL-4B-Instruct-Action> \
  --save-dir  <Qwen3-VL-4B-Instruct-ActionEffect> \
  --tokenizer-ckpt <stage1_run>/checkpoints/effect_tokenizer.pt

# Stage 2 — train QwenEffect (effect-conditioned FAST action prediction)
bash examples/ActionEffect/train_files/run_stage2_qweneffect.sh
```

## 5. Stage-2 token layout

The assistant target string becomes a **latent plan → action** chain (CoT-style):

```
<effect_12><effect_531> ... (N×L effect tokens)   <robot_action_98><robot_action_1203>...
└──── RVQ effect plan (id = level*C + code) ────┘  └──────── FAST action tokens ────────┘
```

(In `execution_mode: flow`, only the effect plan is emitted as tokens; the action chunk is
produced by the continuous flow head instead of FAST tokens.)

`QwenEffect` computes the *ground-truth* effect tokens on the fly with the frozen Stage-1
encoder (it needs future frames, which the `EffectLibero` data config provides only during
training). At inference the VLM **generates** the effect tokens first, then the action tokens —
no future frames required. A `decode_mode="decoder"` option instead decodes actions with the
frozen Stage-1 `D_a` (the XR-1/UniT style "predict codes, decode separately" path).

## 6. Files

```
effect_tokenizer/
  feature_extractor.py   frozen visual (DINOv2/SigLIP + Qwen-VL tower) + text encoders
  vq.py                  VectorQuantizer + ResidualVQ (multi-level), no extra deps
  model.py               EffectAwareTokenizer: encoders, decoders, losses (+L_align), encode→codes
  flow_head.py           optional conditional flow-matching executor (execution_mode=flow)
  dataset.py             LIBERO LeRobot dataset that also returns o_{t+H} (future obs)
  train_tokenizer.py     Stage-1 training loop (accelerate + deepspeed-compatible)
  export_effect_tokens.py codebook usage stats + per-sample code dump
train_files/
  data_registry/data_config.py   EffectLibero data config (future frames) + mixtures
  add_effect_tokens.py            extend Qwen vocab with <effect_0..C-1>
  starvla_effect_libero.yaml      Stage-2 config
  run_stage1_tokenizer.sh         Stage-1 launcher
  run_stage2_qweneffect.sh        Stage-2 launcher
```

The Stage-2 framework itself lives in the package so the registry can auto-discover it:

```
starVLA/model/framework/VLM4A/QwenEffect.py   thin framework; lazily imports this folder
```

> One small enabling hook lives outside this folder:
> `starVLA/model/framework/VLM4A/QwenEffect.py` (auto-registers the Stage-2 framework and
> lazily imports the heavy logic from here), plus a guarded `include_future_obs` branch in
> `starVLA/dataloader/gr00t_lerobot/datasets.py::_pack_sample` (default off, behaviour
> unchanged for every other config).

## 7. wall-oss-inspired upgrades (now implemented)

After comparing with **WALL-OSS-0.5** ([arXiv 2605.30877](https://arxiv.org/html/2605.30877v2),
*Vision-Aligned RVQ Action Tokenizer*), three of its design choices were folded in:

1. **Residual VQ (multi-level).** `vq.py::ResidualVQ` stacks `num_quantizers` codebooks
   (coarse motion → fine residual), like wall-oss's RVQ. Codes are now shaped `(B, N, L)`;
   a (level `l`, code `c`) maps to one flat vocab id `l*num_codes + c`, so the VLM emits
   `N*L` effect tokens. Set `--num_quantizers 1` to fall back to plain VQ.
2. **Visual-action alignment** (`L_align`, weight `lambda_align`). An InfoNCE term pulls the
   pre-VQ effect latent toward a (frozen) visual feature of `o_t`, so the codes become a
   *semantic, visually-grounded* interface for the backbone — wall-oss's key rationale. To
   make it literally **VLM-native**, set `visual_backend=qwen3vl` (uses the Qwen-VL vision
   tower via `feature_extractor.FrozenVLMVisionEncoder`) or point `visual_model_id` at the
   matching SigLIP checkpoint.
3. **Optional continuous executor** (`execution_mode: flow`). wall-oss showed discrete decode
   is "too coarse for precise control" and executes via flow matching while the discrete
   tokens shape the backbone. `effect_tokenizer/flow_head.py::ConditionalFlowHead` (linear
   path `x_t=t·x+(1-t)ε`, velocity + action-space supervision) is conditioned on the VLM
   hidden state; in `flow` mode the effect tokens remain the predicted *plan* (CE loss) and
   the flow head produces the executed actions. Default stays `fast` (pure-AR FAST tokens),
   which QwenFast already validates on LIBERO.

## 8. Caveats

- This is a *design-complete scaffold*. It needs the cluster runtime (GPU, the
  `Qwen3-VL-*-Action` checkpoint, LIBERO LeRobot data, the `starVLA` conda env) to actually
  train; it has not been executed here.
- Default hyper-parameters (`C=512, N=4, H=8`, loss weights) are sensible starting points,
  not tuned numbers.
- The frozen feature/text encoders default to small HF models and fall back to a deterministic
  random projection when offline so the code imports/run-smoke-tests without network; swap in a
  real DINOv2/SigLIP for meaningful effect features.
