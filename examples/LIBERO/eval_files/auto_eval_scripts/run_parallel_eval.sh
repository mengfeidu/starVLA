#!/bin/bash
# Parallel LIBERO eval for this cluster.
# Launches one process per (task_suite -> GPU) pair, all running concurrently.
# Each process owns its own policy server + libero env on a dedicated GPU/port.
#
# Override defaults via env vars, e.g.:
#   MAX_TASKS=1 NUM_TRIALS=2 bash run_parallel_eval.sh   # quick smoke test
#   bash run_parallel_eval.sh                            # full run
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/run_eval_job.sh"

# 5-epoch checkpoint (save_interval 2137 ~ 1 epoch => steps_10685 = epoch 5).
CKPT="${CKPT:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints/0608_qwenfast_libero_all_10ep_trail/checkpoints/steps_10685_pytorch_model.pt}"

# One task suite per GPU (model was trained on libero_all = all 4 suites).
TASK_SUITES=(${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10})
GPU_LIST=(${GPU_LIST:-0 1 2 3})
BASE_PORT="${BASE_PORT:-6700}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS="${MAX_TASKS:--1}"

num_gpus=${#GPU_LIST[@]}

echo "=========================================="
echo " Parallel LIBERO eval"
echo " ckpt        : $CKPT"
echo " task suites : ${TASK_SUITES[*]}"
echo " gpus        : ${GPU_LIST[*]}"
echo " num_trials  : $NUM_TRIALS    max_tasks: $MAX_TASKS"
echo "=========================================="

pids=()
for i in "${!TASK_SUITES[@]}"; do
    task="${TASK_SUITES[$i]}"
    gpu="${GPU_LIST[$((i % num_gpus))]}"
    port=$((BASE_PORT + i))
    echo "[launch] task=$task gpu=$gpu port=$port"
    bash "$JOB" "$CKPT" "$task" "$gpu" "$port" "$NUM_TRIALS" "$MAX_TASKS" &
    pids+=($!)
    sleep 3
done

echo "--- ${#pids[@]} jobs launched; waiting for completion ---"
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo "=========================================="
echo " All parallel eval jobs finished (fail=$fail)"
echo " Results under: playground/eval_results/"
echo "=========================================="
exit $fail
