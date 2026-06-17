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
# Backward-compatible full-training entry point.
export DATA_MIX="${DATA_MIX:-robocasa365_target_human_all}"
export RUN_ID="${RUN_ID:-qwenfast_robocasa365_target_human_all_qwen3vl4b_action}"

export BATCH_SIZE=8
export MAX_TRAIN_STEPS=200000
export SAVE_INTERVAL=10000
export WANDB_MODE=offline
export MAIN_PROCESS_PORT=39746
bash "${SCRIPT_DIR}/run_robocasa365.sh"
