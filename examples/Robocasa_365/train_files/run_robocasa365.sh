#!/usr/bin/env bash
set -eo pipefail

CONDA_BASE="${CONDA_BASE:-/aifs4su/hansirui_4th/miniconda3}"
CONDA_ENV="${CONDA_ENV:-starVLA}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"


# NCCL defaults for the cluster. Override or unset from the environment if needed.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

################################################################################
# Paths and training knobs. Override any value from the command line, e.g.:
#   NUM_PROCESSES=4 BATCH_SIZE=4 DATA_MIX=robocasa365_open_drawer_target_human \
#     bash examples/Robocasa_365/train_files/run_robocasa365.sh
################################################################################
FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenFast}"
BASE_VLM="${BASE_VLM:-/aifs4su/hansirui_4th/ckpts/Qwen3-VL-4B-Instruct-Action}"
CONFIG_YAML="${CONFIG_YAML:-./examples/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml}"
ROBOCASA365_DATA_ROOT="${ROBOCASA365_DATA_ROOT:-/aifs4su/hansirui_4th/dumengfei/benchmark/robocasa/datasets}"
DATA_MIX="${DATA_MIX:-robocasa365_target_human_all}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/aifs4su/hansirui_4th/dumengfei/experiments/starVLA}"
RUN_ID="${RUN_ID:-qwenfast_robocasa365_target_human_all_qwen3vl4b_action}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
# Warmup defaults to the first 10% of total steps; override via NUM_WARMUP_STEPS.
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-$(( MAX_TRAIN_STEPS / 10 ))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
ACTION_DIM="${ACTION_DIM:-12}"
STATE_DIM="${STATE_DIM:-16}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/aifs4su/hansirui_4th/ckpts/fast}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_RoboCasa365_QwenFast}"
WANDB_ENTITY="${WANDB_ENTITY:-dumengfei}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39846}"

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES="$(nvidia-smi -L | wc -l)"
  else
    NUM_PROCESSES=1
  fi
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

echo "Training ${FRAMEWORK_NAME} on ${DATA_MIX}"
echo "  base_vlm: ${BASE_VLM}"
echo "  data_root: ${ROBOCASA365_DATA_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  num_processes: ${NUM_PROCESSES}"
echo "  action_dim: ${ACTION_DIM}  action_horizon: ${ACTION_HORIZON}"
echo "  max_train_steps: ${MAX_TRAIN_STEPS}  warmup_steps: ${NUM_WARMUP_STEPS} (10%)"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --framework.action_model.action_horizon "${ACTION_HORIZON}" \
  --framework.action_model.action_dim "${ACTION_DIM}" \
  --framework.action_model.state_dim "${STATE_DIM}" \
  --framework.action_model.fast_tokenizer_path "${FAST_TOKENIZER_PATH}" \
  --datasets.vla_data.data_root_dir "${ROBOCASA365_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
