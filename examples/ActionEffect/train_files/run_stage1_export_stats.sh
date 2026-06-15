#!/usr/bin/env bash
# Diagnostics: export codebook usage stats for a trained Stage-1 tokenizer.
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

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/hdfs/data/dumengfei/data/playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all_effect}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints}"
TOKENIZER_RUN_ID="${TOKENIZER_RUN_ID:-action_effect_tok_vq}"
EFFECT_TOKENIZER_CKPT="${EFFECT_TOKENIZER_CKPT:-${RUN_ROOT_DIR}/${TOKENIZER_RUN_ID}/checkpoints/effect_tokenizer.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_BATCHES="${MAX_BATCHES:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUT="${OUT:-${RUN_ROOT_DIR}/${TOKENIZER_RUN_ID}/effect_token_stats.json}"

python examples/ActionEffect/effect_tokenizer/export_effect_tokens.py \
  --data_root_dir "${LIBERO_DATA_ROOT}" \
  --data_mix "${DATA_MIX}" \
  --tokenizer-ckpt "${EFFECT_TOKENIZER_CKPT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_batches "${MAX_BATCHES}" \
  --out "${OUT}"
