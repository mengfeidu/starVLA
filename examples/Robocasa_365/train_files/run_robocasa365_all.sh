#!/usr/bin/env bash
set -euo pipefail

source /aifs4su/hansirui_4th/miniconda3/bin/activate
conda activate starVLA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Backward-compatible full-training entry point.
export DATA_MIX="${DATA_MIX:-robocasa365_target_human_all}"
export RUN_ID="${RUN_ID:-qwenfast_robocasa365_target_human_all_qwen3vl4b_action}"

export BATCH_SIZE=16
export MAX_TRAIN_STEPS=200000
export SAVE_INTERVAL=10000
export WANDB_MODE=offline
exec bash "${SCRIPT_DIR}/run_robocasa365.sh"
