#!/bin/bash
# Fast parallel LIBERO eval: shard each task suite across many client processes
# that share ONE policy server per GPU.
#
# Why: with OSMesa (CPU) rendering the bottleneck is CPU, not GPU. A single
# (server+env) pair leaves the GPU ~idle and uses ~9GB/80GB VRAM. This node has
# 122 CPU cores, so we run many env-client processes in parallel. The trained
# model is suite-agnostic (libero_all), so one server per GPU can serve all the
# client shards assigned to that GPU; inference serializes on the GPU (fast)
# while each client renders on CPU in parallel.
#
# Layout:
#   - 1 server per GPU            (loads the 9GB model once per GPU)
#   - SHARDS_PER_SUITE clients per suite, pinned to that suite's GPU server
#   - total clients = num_suites * SHARDS_PER_SUITE
#
# Override via env vars, e.g.:
#   SHARDS_PER_SUITE=5 bash run_parallel_eval_fast.sh        # 5 clients/suite
#   NUM_TRIALS=5 MAX_TASKS=2 bash run_parallel_eval_fast.sh  # quick check
set -u

# ---------------- cluster paths ----------------
STARVLA_DIR=/opt/tiger/robot_policy/starVLA
LIBERO_HOME=/mnt/bn/ic-vlm/personal/dumengfei/benchmarks/LIBERO
STARVLA_PY=/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/envs/starVLA/bin/python
LIBERO_PY=/mnt/bn/ic-vlm/personal/dumengfei/packages/anaconda3/envs/libero/bin/python

cd "$STARVLA_DIR" || exit 1

export PYTHONPATH="$STARVLA_DIR:$LIBERO_HOME:${PYTHONPATH:-}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
# Keep per-process CPU thread fan-out modest so many parallel renders coexist.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# ---------------- config ----------------
CKPT="${CKPT:-/mnt/hdfs/data/dumengfei/data/playground/Checkpoints/0608_qwenfast_libero_all_10ep_trail/checkpoints/steps_10685_pytorch_model.pt}"
TASK_SUITES=(${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10})
GPU_LIST=(${GPU_LIST:-0 1 2 3})
BASE_PORT="${BASE_PORT:-6700}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS="${MAX_TASKS:--1}"          # cap tasks per suite (for quick checks); -1 = all
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-5}"
# Model servers per GPU. The model is small (~9GB) vs 80GB VRAM, and a single
# asyncio server serializes inference, so >1 server per GPU adds inference
# parallelism and raises GPU/CPU utilization. Clients are round-robined across
# ALL servers (model is suite-agnostic), so no suite->GPU affinity is needed.
SERVERS_PER_GPU="${SERVERS_PER_GPU:-1}"

num_gpus=${#GPU_LIST[@]}
host=127.0.0.1

# standard task counts; fallback 10
suite_ntasks() {
    case "$1" in
        libero_spatial|libero_object|libero_goal|libero_10) echo 10 ;;
        libero_90) echo 90 ;;
        libero_100) echo 130 ;;
        *) echo 10 ;;
    esac
}

# ---------------- self-heal: OSMesa + LIBERO config ----------------
if ! ldconfig -p 2>/dev/null | grep -qi "libOSMesa"; then
    echo "[setup] installing libosmesa6 via apt"
    sudo apt-get install -y libosmesa6 >/tmp/osmesa_install.log 2>&1
fi
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    root="$LIBERO_HOME/libero/libero"
    cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
assets: $root/assets
bddl_files: $root/bddl_files
benchmark_root: $root
datasets: $root/../datasets
init_states: $root/init_files
EOF
    echo "[setup] wrote LIBERO config -> $LIBERO_CONFIG_PATH/config.yaml"
fi

# ---------------- output paths ----------------
folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
out_root="${STARVLA_DIR}/playground/eval_results/${folder_name}"
log_path="${out_root}/logs"
res_path="${out_root}/results"
mkdir -p "$log_path" "$res_path"

echo "=========================================="
echo " Fast parallel LIBERO eval"
echo " ckpt            : $CKPT"
echo " task suites     : ${TASK_SUITES[*]}"
echo " gpus            : ${GPU_LIST[*]}  (${SERVERS_PER_GPU} server(s) each)"
echo " shards/suite    : $SHARDS_PER_SUITE   num_trials: $NUM_TRIALS   max_tasks: $MAX_TASKS"
echo " total servers   : $(( num_gpus * SERVERS_PER_GPU ))"
echo " total clients   : $(( ${#TASK_SUITES[@]} * SHARDS_PER_SUITE ))"
echo "=========================================="

# ---------------- 1) start SERVERS_PER_GPU servers on each GPU ----------------
server_pids=()
server_ports=()
sidx=0
for ((r=0; r<SERVERS_PER_GPU; r++)); do
    for i in "${!GPU_LIST[@]}"; do
        gpu="${GPU_LIST[$i]}"
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

# ---------------- 2) wait for servers to be ready (port opens only after model load) ----------------
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

# ---------------- 3) launch client shards, round-robin across ALL servers ----------------
num_servers=${#server_ports[@]}
client_pids=()
cidx=0
for si in "${!TASK_SUITES[@]}"; do
    suite="${TASK_SUITES[$si]}"
    ntasks=$(suite_ntasks "$suite")
    if [ "$MAX_TASKS" -gt 0 ] && [ "$MAX_TASKS" -lt "$ntasks" ]; then ntasks="$MAX_TASKS"; fi

    nshards="$SHARDS_PER_SUITE"
    [ "$nshards" -gt "$ntasks" ] && nshards="$ntasks"

    for ((s=0; s<nshards; s++)); do
        # round-robin task ids into this shard: task t -> shard (t % nshards)
        ids=""
        for ((t=0; t<ntasks; t++)); do
            if [ $((t % nshards)) -eq "$s" ]; then ids="${ids:+$ids,}$t"; fi
        done
        [ -z "$ids" ] && continue
        port="${server_ports[$((cidx % num_servers))]}"
        tag="${suite}_shard${s}"
        rj="${res_path}/${tag}.json"
        clog="${log_path}/${tag}.log"
        echo "[client] $tag -> server port $port, task_ids=$ids"
        "$LIBERO_PY" ./examples/LIBERO/eval_files/eval_libero.py \
            --args.host "$host" \
            --args.port "$port" \
            --args.task-suite-name "$suite" \
            --args.task-ids "$ids" \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "${out_root}/videos/${suite}" \
            --args.result-json "$rj" \
            < /dev/null > "$clog" 2>&1 &
        client_pids+=($!)
        cidx=$((cidx + 1))
        sleep 1
    done
done

echo "--- ${#client_pids[@]} client shards launched; waiting for completion ---"
fail=0
for pid in "${client_pids[@]}"; do
    wait "$pid" || fail=1
done

# ---------------- 4) aggregate per-suite success rates ----------------
echo "=========================================="
echo " Aggregated results"
echo "=========================================="
"$LIBERO_PY" - "$res_path" <<'PYEOF'
import json, sys, glob, os
res_dir = sys.argv[1]
suites = {}
for fp in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    if os.path.basename(fp) == "aggregate.json":
        continue
    with open(fp) as f:
        d = json.load(f)
    s = d["task_suite_name"]
    e, c = suites.get(s, (0, 0))
    suites[s] = (e + d["total_episodes"], c + d["total_successes"])
agg = {}
tot_e = tot_c = 0
for s in sorted(suites):
    e, c = suites[s]
    sr = (c / e * 100) if e else 0.0
    agg[s] = {"episodes": e, "successes": c, "success_rate": round(sr, 2)}
    tot_e += e; tot_c += c
    print(f"  {s:16s}: {c}/{e} = {sr:.2f}%")
overall = (tot_c / tot_e * 100) if tot_e else 0.0
print(f"  {'AVG/overall':16s}: {tot_c}/{tot_e} = {overall:.2f}%")
agg["__overall__"] = {"episodes": tot_e, "successes": tot_c, "success_rate": round(overall, 2)}
with open(os.path.join(res_dir, "aggregate.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(f"\n  wrote {os.path.join(res_dir, 'aggregate.json')}")
PYEOF

# ---------------- 5) upload results to the checkpoint's hdfs dir ----------------
# Maps the local FUSE path (/mnt/hdfs/data/dumengfei/...) to its hdfs:// URL and
# uploads eval_results under the checkpoint dir, e.g.
#   hdfs://.../Checkpoints/<run_id>/eval_results/<folder_name>/{logs,videos,results}
if [ "${UPLOAD_TO_HDFS:-1}" != "0" ] && command -v hdfs >/dev/null 2>&1; then
    model_root="$(dirname "$(dirname "$CKPT")")"   # .../Checkpoints/<run_id>
    HDFS_LOCAL_PREFIX="${HDFS_LOCAL_PREFIX:-/mnt/hdfs/data/dumengfei}"
    HDFS_URL_PREFIX="${HDFS_URL_PREFIX:-hdfs://haruna/home/byte_data_seed/hdd_hldy/iccv/user/dumengfei}"
    if [[ "$model_root" != "$HDFS_LOCAL_PREFIX"* ]]; then
        echo "[upload] skip: $model_root is not under $HDFS_LOCAL_PREFIX (set HDFS_LOCAL_PREFIX/HDFS_URL_PREFIX)"
    else
        hdfs_dest="${model_root/#$HDFS_LOCAL_PREFIX/$HDFS_URL_PREFIX}/eval_results"
        echo "[upload] $out_root -> $hdfs_dest/"
        hdfs dfs -mkdir -p "$hdfs_dest" 2>/dev/null
        hdfs dfs -rm -r -f "$hdfs_dest/$(basename "$out_root")" >/dev/null 2>&1
        if hdfs dfs -put -f "$out_root" "$hdfs_dest/"; then
            echo "[upload] OK -> $hdfs_dest/$(basename "$out_root")"
            if [ "${KEEP_LOCAL_RESULTS:-0}" = "0" ]; then
                rm -rf "$out_root"
                echo "[upload] removed local $out_root (set KEEP_LOCAL_RESULTS=1 to keep)"
            fi
        else
            echo "[upload] FAILED; local results kept at $out_root"
        fi
    fi
fi

echo "=========================================="
echo " Done (client fail flag=$fail). Servers will be stopped now."
echo " Logs:    $log_path"
echo " Results: $res_path/aggregate.json"
echo "=========================================="
exit $fail
