#!/usr/bin/env bash
# Stage 1 precompute: cache frozen Qwen-ViT features for Effect Tokenizer training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/bin/activate}"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  source "${CONDA_ACTIVATE}"
  conda activate "${CONDA_ENV_NAME:-starVLA}"
else
  echo "Conda activate script not found at ${CONDA_ACTIVATE}; using current environment."
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/hdfs/data/dumengfei/data/playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_plus_90_effect}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
FEATURE_ROOT_DIR="${FEATURE_ROOT_DIR:-/mnt/hdfs/data/dumengfei/data/playground/Features}"
FEATURE_ID="${FEATURE_ID:-qwen_vit_${DATA_MIX}}"
OUTPUT_DIR="${FEATURE_ROOT_DIR}/${FEATURE_ID}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHARD_SIZE="${SHARD_SIZE:-8192}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
SAVE_DELTA="${SAVE_DELTA:-0}"
FAIL_ON_BAD_SAMPLE="${FAIL_ON_BAD_SAMPLE:-0}"
MAX_BAD_SAMPLE_RATIO="${MAX_BAD_SAMPLE_RATIO:-0.05}"
BAD_SAMPLE_CHECK_MIN="${BAD_SAMPLE_CHECK_MIN:-512}"
LOG_EVERY="${LOG_EVERY:-20}"
VISUAL_BACKEND="${VISUAL_BACKEND:-qwen3vl}"
VISUAL_MODEL_ID="${VISUAL_MODEL_ID:-/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

RESOLVED_LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}"
if [[ -d "${LIBERO_DATA_ROOT}/LEROBOT_LIBERO_DATA" && ! -d "${LIBERO_DATA_ROOT}/libero_object_no_noops_1.0.0_lerobot" ]]; then
  RESOLVED_LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT}/LEROBOT_LIBERO_DATA"
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then NUM_PROCESSES="$(nvidia-smi -L | wc -l)"; else NUM_PROCESSES=1; fi
fi

mkdir -p "${OUTPUT_DIR}"
echo "Cache Qwen-ViT features | data=${RESOLVED_LIBERO_DATA_ROOT} mix=${DATA_MIX} out=${OUTPUT_DIR}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

SAVE_DELTA_ARGS=()
if [[ "${SAVE_DELTA}" == "1" ]]; then
  SAVE_DELTA_ARGS+=(--save_delta)
fi

BAD_SAMPLE_ARGS=()
if [[ "${FAIL_ON_BAD_SAMPLE}" == "1" ]]; then
  BAD_SAMPLE_ARGS+=(--fail_on_bad_sample)
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-39849}" \
  examples/ActionEffect/effect_tokenizer/cache_qwen_vit_features.py \
  --data_root_dir "${RESOLVED_LIBERO_DATA_ROOT}" \
  --data_mix "${DATA_MIX}" \
  --video_backend "${VIDEO_BACKEND}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --shard_size "${SHARD_SIZE}" \
  --save_dtype "${SAVE_DTYPE}" \
  --log_every "${LOG_EVERY}" \
  --max_bad_sample_ratio "${MAX_BAD_SAMPLE_RATIO}" \
  --bad_sample_check_min "${BAD_SAMPLE_CHECK_MIN}" \
  --visual_backend "${VISUAL_BACKEND}" \
  --visual_model_id "${VISUAL_MODEL_ID}" \
  --image_size "${IMAGE_SIZE}" \
  "${SAVE_DELTA_ARGS[@]}" \
  "${BAD_SAMPLE_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
