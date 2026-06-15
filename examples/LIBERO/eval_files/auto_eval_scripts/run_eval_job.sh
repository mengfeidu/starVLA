#!/bin/bash
# Single LIBERO eval job for this cluster.
#   - Launches the starVLA policy server (starVLA conda env) on one GPU + port.
#   - Runs the LIBERO eval client (libero conda env) against it.
#   - Cleans up the server on exit.
#
# Self-healing: this node is an ephemeral container whose root FS (apt packages,
# repo, $HOME) can be reset between sessions, while /mnt/* persists. So each run
# (re)installs OSMesa if missing and regenerates the LIBERO config if missing.
#
# Usage:
#   bash run_eval_job.sh <ckpt> <task_suite> <gpu_id> <port> [num_trials] [max_tasks]
set -u

# ---------------- args ----------------
CKPT="$1"          # .../checkpoints/steps_XXXX_pytorch_model.pt
TASK_SUITE="$2"    # libero_spatial | libero_object | libero_goal | libero_10
GPU_ID="$3"
PORT="$4"
NUM_TRIALS="${5:-50}"
MAX_TASKS="${6:--1}"

# ---------------- cluster paths ----------------
STARVLA_DIR=/opt/tiger/robot_policy/starVLA
LIBERO_HOME=/mnt/bn/ic-vlm/personal/dumengfei/benchmarks/LIBERO
STARVLA_PY=/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/envs/starVLA/bin/python
LIBERO_PY=/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/envs/libero/bin/python

cd "$STARVLA_DIR" || exit 1

export PYTHONPATH="$STARVLA_DIR:$LIBERO_HOME:${PYTHONPATH:-}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
# CPU (software) rendering via OSMesa: this node has no system libEGL, so the
# GPU EGL backend is unavailable.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

host=127.0.0.1

# ---------------- self-heal: OSMesa (apt) ----------------
# Guarded by a lock so parallel jobs don't race on apt.
ensure_osmesa() {
    if ldconfig -p 2>/dev/null | grep -qi "libOSMesa"; then
        return 0
    fi
    echo "[job] libOSMesa missing -> installing libosmesa6 via apt"
    sudo apt-get install -y libosmesa6 >/tmp/osmesa_install.log 2>&1
}
(
    flock 9
    ensure_osmesa
) 9>/tmp/osmesa_install.lock

# ---------------- self-heal: LIBERO config ----------------
# LIBERO prompts interactively on first import if config.yaml is missing.
# Generate it non-interactively from the (persistent) LIBERO repo paths.
ensure_libero_config() {
    local cfg_dir="$LIBERO_CONFIG_PATH"
    local cfg="$cfg_dir/config.yaml"
    [ -f "$cfg" ] && return 0
    mkdir -p "$cfg_dir"
    local root="$LIBERO_HOME/libero/libero"
    cat > "$cfg" <<EOF
assets: $root/assets
bddl_files: $root/bddl_files
benchmark_root: $root
datasets: $root/../datasets
init_states: $root/init_files
EOF
    echo "[job] wrote LIBERO config -> $cfg"
}
ensure_libero_config

# ---------------- output paths (local disk to avoid hdfs write churn) ----------------
folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
out_root="${STARVLA_DIR}/playground/eval_results/${folder_name}"
video_out_path="${out_root}/videos/${TASK_SUITE}"
log_path="${out_root}/logs"
mkdir -p "$video_out_path" "$log_path"
client_log="${log_path}/${TASK_SUITE}.log"
server_log="${log_path}/server_${TASK_SUITE}.log"

echo "[job] GPU=$GPU_ID port=$PORT task=$TASK_SUITE trials=$NUM_TRIALS max_tasks=$MAX_TASKS" | tee "$client_log"
echo "[job] ckpt=$CKPT" | tee -a "$client_log"

# ---------------- start policy server (starVLA env) ----------------
CUDA_VISIBLE_DEVICES="$GPU_ID" "$STARVLA_PY" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" \
    --port "$PORT" \
    --use_bf16 > "$server_log" 2>&1 &
server_pid=$!
echo "[job] server pid=$server_pid (log: $server_log)" | tee -a "$client_log"

cleanup() { kill "$server_pid" 2>/dev/null; }
trap cleanup EXIT

# ---------------- run eval client (libero env) ----------------
# Client retries connection to the server for up to 300s, so no manual wait needed.
"$LIBERO_PY" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.host "$host" \
    --args.port "$PORT" \
    --args.task-suite-name "$TASK_SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.max-tasks "$MAX_TASKS" \
    --args.video-out-path "$video_out_path" \
    < /dev/null 2>&1 | tee -a "$client_log"

echo "[job] finished task=$TASK_SUITE (videos: $video_out_path)" | tee -a "$client_log"
