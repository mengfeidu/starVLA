#!/usr/bin/env bash
# Stage 1: train the action-first Effect Tokenizer on LIBERO LeRobot data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Runtime environment (same cluster setup as the LIBERO QwenFast script when available).
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/bin/activate}"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  source "${CONDA_ACTIVATE}"
  conda activate "${CONDA_ENV_NAME:-starVLA}"
else
  echo "Conda activate script not found at ${CONDA_ACTIVATE}; using current environment."
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/hdfs/data/dumengfei/data/playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_effect}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints}"
RUN_ID="${RUN_ID:-action_effect_tok_vq}"
OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-40000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
LR="${LR:-2e-4}"
NUM_EFFECT_TOKENS="${NUM_EFFECT_TOKENS:-4}"
NUM_CODES="${NUM_CODES:-512}"
NUM_QUANTIZERS="${NUM_QUANTIZERS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
VISUAL_BACKEND="${VISUAL_BACKEND:-qwen3vl}"   # qwen3vl | dino | auto
VISUAL_MODEL_ID="${VISUAL_MODEL_ID:-/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
TEXT_MODEL_ID="${TEXT_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
LAMBDA_VISUAL="${LAMBDA_VISUAL:-1.0}"
LAMBDA_COMMIT="${LAMBDA_COMMIT:-1.0}"
VQ_BETA="${VQ_BETA:-0.1}"
VQ_EMA_DECAY="${VQ_EMA_DECAY:-0.99}"
VQ_DEAD_CODE_THRESHOLD="${VQ_DEAD_CODE_THRESHOLD:-1.0}"
NO_VQ_EMA="${NO_VQ_EMA:-0}"

RESOLVED_LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}"
if [[ -d "${LIBERO_DATA_ROOT}/LEROBOT_LIBERO_DATA" && ! -d "${LIBERO_DATA_ROOT}/libero_object_no_noops_1.0.0_lerobot" ]]; then
  RESOLVED_LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}/LEROBOT_LIBERO_DATA"
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then NUM_PROCESSES="$(nvidia-smi -L | wc -l)"; else NUM_PROCESSES=1; fi
fi

mkdir -p "${OUTPUT_DIR}"
echo "Stage1 action-first effect tokenizer | data=${RESOLVED_LIBERO_DATA_ROOT} mix=${DATA_MIX} out=${OUTPUT_DIR}"

VQ_EMA_ARGS=()
if [[ "${NO_VQ_EMA}" == "1" ]]; then
  VQ_EMA_ARGS+=(--no_vq_ema)
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-39847}" \
  examples/ActionEffect/effect_tokenizer/train_tokenizer.py \
  --data_root_dir "${RESOLVED_LIBERO_DATA_ROOT}" \
  --data_mix "${DATA_MIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --target_epochs "${TARGET_EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --lr "${LR}" \
  --action_horizon "${ACTION_HORIZON}" \
  --num_effect_tokens "${NUM_EFFECT_TOKENS}" \
  --num_codes "${NUM_CODES}" \
  --num_quantizers "${NUM_QUANTIZERS}" \
  --visual_backend "${VISUAL_BACKEND}" \
  --visual_model_id "${VISUAL_MODEL_ID}" \
  --image_size "${IMAGE_SIZE}" \
  --text_model_id "${TEXT_MODEL_ID}" \
  --lambda_action "${LAMBDA_ACTION}" \
  --lambda_visual "${LAMBDA_VISUAL}" \
  --lambda_commit "${LAMBDA_COMMIT}" \
  --vq_beta "${VQ_BETA}" \
  --vq_ema_decay "${VQ_EMA_DECAY}" \
  --vq_dead_code_threshold "${VQ_DEAD_CODE_THRESHOLD}" \
  "${VQ_EMA_ARGS[@]}" \
  --lambda_lang 0.0 \
  --lambda_inverse 0.0 \
  --lambda_align 0.0
