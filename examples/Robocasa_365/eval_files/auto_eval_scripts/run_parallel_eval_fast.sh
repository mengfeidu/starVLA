#!/bin/bash
# Fast parallel RoboCasa365 eval: shard target tasks across many client processes
# that share one or more policy servers.
#
# Layout:
#   - SERVERS_PER_GPU policy servers per GPU
#   - SHARDS clients, each evaluating a round-robin subset of the 50 target tasks
#   - clients are round-robined across all policy servers
#
# Override via env vars, e.g.:
#   SHARDS=10 NUM_EPISODES=5 bash run_parallel_eval_fast.sh
#   TASK_IDS=0,6,18 SHARDS=3 bash run_parallel_eval_fast.sh
set -u

# ---------------- cluster paths ----------------
STARVLA_DIR="${STARVLA_DIR:-/aifs4su/hansirui_4th/dumengfei/code/starVLA}"
STARVLA_PY="${STARVLA_PY:-/aifs4su/hansirui_4th/miniconda3/envs/starVLA/bin/python}"
ROBOCASA_PY="${ROBOCASA_PY:-/aifs4su/hansirui_4th/miniconda3/envs/robocasa365/bin/python}"

cd "$STARVLA_DIR" || exit 1

export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export MUJOCO_GL="$RENDER_BACKEND"
export PYOPENGL_PLATFORM="$RENDER_BACKEND"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# ---------------- config ----------------
CKPT="${CKPT:-/aifs4su/hansirui_4th/dumengfei/experiments/starVLA/qwenfast_robocasa365_target_human_all_qwen3vl4b_action/final_model/pytorch_model.pt}"
GPU_LIST=(${GPU_LIST:-0})
BASE_PORT="${BASE_PORT:-6800}"
NUM_EPISODES="${NUM_EPISODES:-5}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
SHARDS="${SHARDS:-5}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-1}"
TASK_IDS="${TASK_IDS:-}"          # comma/space-separated task ids; empty = all 50 target tasks
UPLOAD_TO_HDFS="${UPLOAD_TO_HDFS:-0}"

# ---------------- setup checks ----------------
case "$RENDER_BACKEND" in
    egl)
        if ! ldconfig -p 2>/dev/null | grep -qi "libEGL"; then
            echo "[setup] ERROR: libEGL is required for RENDER_BACKEND=egl but was not found."
            echo "[setup] Install EGL/NVIDIA GL libraries, or rerun with RENDER_BACKEND=osmesa."
            exit 1
        fi
        ;;
    osmesa)
        ROBOCASA_ENV_PREFIX="$(dirname "$(dirname "$ROBOCASA_PY")")"
        if ls "${ROBOCASA_ENV_PREFIX}/lib"/libOSMesa.so* >/dev/null 2>&1; then
            export LD_LIBRARY_PATH="${ROBOCASA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        elif ! ldconfig -p 2>/dev/null | grep -qi "libOSMesa"; then
            echo "[setup] ERROR: libOSMesa is required for RENDER_BACKEND=osmesa but was not found."
            exit 1
        fi
        ;;
    *)
        echo "[setup] ERROR: unsupported RENDER_BACKEND=$RENDER_BACKEND (expected egl or osmesa)."
        exit 1
        ;;
esac

resolve_task_ids() {
    if [ -n "$TASK_IDS" ]; then
        echo "$TASK_IDS" | tr ',' ' '
    else
        seq 0 49
    fi
}

all_task_ids=($(resolve_task_ids))
if [ "${#all_task_ids[@]}" -eq 0 ]; then
    echo "[setup] ERROR: no task ids resolved"
    exit 1
fi

if [ "$SHARDS" -gt "${#all_task_ids[@]}" ]; then
    SHARDS="${#all_task_ids[@]}"
fi

# ---------------- output paths ----------------
model_root="$(dirname "$(dirname "$CKPT")")"
ckpt_dir_name="$(basename "$(dirname "$CKPT")")"
ckpt_file_name="$(basename "$CKPT")"
folder_name="${ckpt_dir_name}_${ckpt_file_name%.*}"
out_root="${EVAL_ROOT:-${model_root}/eval_results/robocasa365_${folder_name}}"
log_path="${out_root}/logs"
res_path="${out_root}/results"
mkdir -p "$log_path" "$res_path"

echo "=========================================="
echo " Fast parallel RoboCasa365 eval"
echo " ckpt            : $CKPT"
echo " output root     : $out_root"
echo " task ids        : ${all_task_ids[*]}"
echo " gpus            : ${GPU_LIST[*]}  (${SERVERS_PER_GPU} server(s) each)"
echo " shards          : $SHARDS   episodes/task: $NUM_EPISODES"
echo " n_envs          : $N_ENVS   max_steps: $MAX_EPISODE_STEPS   n_action_steps: $N_ACTION_STEPS"
echo " total servers   : $(( ${#GPU_LIST[@]} * SERVERS_PER_GPU ))"
echo "=========================================="

# ---------------- 1) start servers ----------------
server_pids=()
server_ports=()
sidx=0
for ((r=0; r<SERVERS_PER_GPU; r++)); do
    for gpu in "${GPU_LIST[@]}"; do
        port=$((BASE_PORT + sidx))
        slog="${log_path}/server_gpu${gpu}_p${port}.log"
        echo "[server] GPU=$gpu port=$port -> $slog"
        CUDA_VISIBLE_DEVICES="$gpu" "$STARVLA_PY" deployment/model_server/server_policy.py \
            --ckpt_path "$CKPT" \
            --port "$port" \
            --use_bf16 \
            --idle_timeout -1 > "$slog" 2>&1 &
        server_pids+=($!)
        server_ports+=($port)
        sidx=$((sidx + 1))
    done
done
cleanup() { echo "[cleanup] stopping servers: ${server_pids[*]}"; kill "${server_pids[@]}" 2>/dev/null; }
trap cleanup EXIT

# ---------------- 2) wait for servers ----------------
wait_port() {
    local p="$1" tries=0
    while ! (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -gt 360 ]; then echo "[wait] server on port $p not ready after 1800s"; return 1; fi
        sleep 5
    done
    exec 3>&- 2>/dev/null
    return 0
}
for port in "${server_ports[@]}"; do
    echo "[wait] waiting for server on port $port ..."
    wait_port "$port" && echo "[wait] server on port $port READY"
done

# ---------------- 3) launch client shards ----------------
num_servers=${#server_ports[@]}
client_pids=()
cidx=0
for ((s=0; s<SHARDS; s++)); do
    ids=""
    for i in "${!all_task_ids[@]}"; do
        if [ $((i % SHARDS)) -eq "$s" ]; then
            ids="${ids:+$ids,}${all_task_ids[$i]}"
        fi
    done
    [ -z "$ids" ] && continue

    port="${server_ports[$((cidx % num_servers))]}"
    tag="target_shard${s}"
    rj="${res_path}/${tag}.json"
    clog="${log_path}/${tag}.log"
    echo "[client] $tag -> server port $port, task_ids=$ids"
    "$ROBOCASA_PY" -m examples.Robocasa_365.eval_files.simulation_env \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-ids "$ids" \
        --args.n-episodes "$NUM_EPISODES" \
        --args.n-envs "$N_ENVS" \
        --args.max-episode-steps "$MAX_EPISODE_STEPS" \
        --args.n-action-steps "$N_ACTION_STEPS" \
        --args.video-out-path "${out_root}/videos/${tag}" \
        --args.result-json "$rj" \
        < /dev/null > "$clog" 2>&1 &
    client_pids+=($!)
    cidx=$((cidx + 1))
    sleep 1
done

echo "--- ${#client_pids[@]} client shards launched; waiting for completion ---"
fail=0
for pid in "${client_pids[@]}"; do
    wait "$pid" || fail=1
done

# ---------------- 4) aggregate results ----------------
echo "=========================================="
echo " Aggregated results"
echo "=========================================="
"$ROBOCASA_PY" - "$res_path" <<'PYEOF'
import json, sys, glob, os

res_dir = sys.argv[1]
task_results = []
total_episodes = 0
total_successes = 0
for fp in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    if os.path.basename(fp) == "aggregate.json":
        continue
    with open(fp) as f:
        d = json.load(f)
    total_episodes += d["total_episodes"]
    total_successes += d["total_successes"]
    task_results.extend(d.get("task_results", []))

by_env = {}
for item in task_results:
    env = item["env"]
    cur = by_env.setdefault(env, {"episodes": 0, "successes": 0})
    cur["episodes"] += item["total_episodes"]
    cur["successes"] += item["total_successes"]

for env in sorted(by_env):
    e = by_env[env]["episodes"]
    c = by_env[env]["successes"]
    sr = (c / e * 100) if e else 0.0
    by_env[env]["success_rate"] = round(sr, 2)
    print(f"  {env:36s}: {c}/{e} = {sr:.2f}%")

overall = (total_successes / total_episodes * 100) if total_episodes else 0.0
print(f"  {'AVG/overall':36s}: {total_successes}/{total_episodes} = {overall:.2f}%")
agg = {
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "success_rate": round(overall, 2),
    "tasks": by_env,
}
out = os.path.join(res_dir, "aggregate.json")
with open(out, "w") as f:
    json.dump(agg, f, indent=2)
print(f"\n  wrote {out}")
PYEOF

# ---------------- 5) optional upload ----------------
if [ "$UPLOAD_TO_HDFS" != "0" ] && command -v hdfs >/dev/null 2>&1; then
    HDFS_LOCAL_PREFIX="${HDFS_LOCAL_PREFIX:-/aifs4su/hansirui_4th/dumengfei/experiments}"
    HDFS_URL_PREFIX="${HDFS_URL_PREFIX:-hdfs://haruna/home/byte_data_seed/hdd_hldy/iccv/user/dumengfei}"
    if [[ "$model_root" != "$HDFS_LOCAL_PREFIX"* ]]; then
        echo "[upload] skip: $model_root is not under $HDFS_LOCAL_PREFIX"
    else
        hdfs_dest="${model_root/#$HDFS_LOCAL_PREFIX/$HDFS_URL_PREFIX}/eval_results"
        echo "[upload] $out_root -> $hdfs_dest/"
        hdfs dfs -mkdir -p "$hdfs_dest" 2>/dev/null
        hdfs dfs -rm -r -f "$hdfs_dest/$(basename "$out_root")" >/dev/null 2>&1
        hdfs dfs -put -f "$out_root" "$hdfs_dest/" || echo "[upload] FAILED; local results kept at $out_root"
    fi
fi

echo "=========================================="
echo " Done (client fail flag=$fail). Servers will be stopped now."
echo " Logs:    $log_path"
echo " Results: $res_path/aggregate.json"
echo "=========================================="
exit $fail
