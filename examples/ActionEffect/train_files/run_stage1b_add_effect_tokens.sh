#!/usr/bin/env bash
# Stage 1b: append <effect_*> tokens to a QwenFast Action checkpoint.
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

RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints}"
TOKENIZER_RUN_ID="${TOKENIZER_RUN_ID:-action_effect_tok_vq}"
EFFECT_TOKENIZER_CKPT="${EFFECT_TOKENIZER_CKPT:-${RUN_ROOT_DIR}/${TOKENIZER_RUN_ID}/checkpoints/effect_tokenizer.pt}"

ACTION_MODEL_ID="${ACTION_MODEL_ID:-/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-Action}"
SAVE_DIR="${SAVE_DIR:-/mnt/hdfs/data/dumengfei/checkpoints/Qwen3-VL-4B-Instruct-ActionEffect}"
INIT_STRATEGY="${INIT_STRATEGY:-normal}"

echo "Stage1b add effect tokens | tokenizer=${EFFECT_TOKENIZER_CKPT}"
echo "  model=${ACTION_MODEL_ID}"
echo "  save=${SAVE_DIR}"

python examples/ActionEffect/train_files/add_effect_tokens.py \
  --model-id "${ACTION_MODEL_ID}" \
  --save-dir "${SAVE_DIR}" \
  --tokenizer-ckpt "${EFFECT_TOKENIZER_CKPT}" \
  --init-strategy "${INIT_STRATEGY}"
