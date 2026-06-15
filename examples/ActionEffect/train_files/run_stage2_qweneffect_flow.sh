#!/usr/bin/env bash
# Stage 2 ablation: train QwenEffect with effect-token CE + continuous flow executor.
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

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenEffect}"
BASE_VLM="${BASE_VLM:-/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-ActionEffect}"
CONFIG_YAML="${CONFIG_YAML:-./examples/ActionEffect/train_files/starvla_effect_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/hdfs/data/dumengfei/data/playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_effect}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints}"
RUN_ID="${RUN_ID:-qweneffect_flow_libero_all}"
TOKENIZER_RUN_ID="${TOKENIZER_RUN_ID:-action_effect_tok_vq}"
EFFECT_TOKENIZER_CKPT="${EFFECT_TOKENIZER_CKPT:-${RUN_ROOT_DIR}/${TOKENIZER_RUN_ID}/checkpoints/effect_tokenizer.pt}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/mnt/hdfs/data/dumengfei/data/playground/Pretrained_models/fast}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-$(( MAX_TRAIN_STEPS / 10 ))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
ACTION_DIM="${ACTION_DIM:-7}"
LAMBDA_FLOW="${LAMBDA_FLOW:-1.0}"
FLOW_SAMPLE_STEPS="${FLOW_SAMPLE_STEPS:-10}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39849}"

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then NUM_PROCESSES="$(nvidia-smi -L | wc -l)"; else NUM_PROCESSES=1; fi
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.action_model.action_horizon "${ACTION_HORIZON}" \
  --framework.action_model.action_dim "${ACTION_DIM}" \
  --framework.action_model.fast_tokenizer_path "${FAST_TOKENIZER_PATH}" \
  --framework.effect.tokenizer_ckpt "${EFFECT_TOKENIZER_CKPT}" \
  --framework.effect.execution_mode flow \
  --framework.effect.lambda_flow "${LAMBDA_FLOW}" \
  --framework.effect.flow_sample_steps "${FLOW_SAMPLE_STEPS}" \
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.include_future_obs true \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --trainer.freeze_modules "effect_tokenizer" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
