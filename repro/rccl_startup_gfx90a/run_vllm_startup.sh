#!/bin/bash
#SBATCH --job-name=rccl-vllm-startup
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --output=rccl-vllm-%j.out
#SBATCH --error=rccl-vllm-%j.err

# Measure real vLLM multi-node startup, phase by phase, and check whether the signature
# found by run_rccl_probe.sh carries into the actual service.
#
#   sbatch repro/rccl_startup_gfx90a/run_vllm_startup.sh                       # 2 nodes, 3 variants
#   MODE=sweep sbatch --nodes=4 repro/rccl_startup_gfx90a/run_vllm_startup.sh
#   MODEL=moonshotai/Kimi-K2-Instruct-0905 PP_SIZE=4 \
#     EXTRA_VLLM_ARGS="--trust-remote-code --quantization fp8 --kv-cache-dtype fp8 \
#     --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 \
#     --gpu-memory-utilization 0.95" \
#     sbatch --nodes=4 --time=04:00:00 repro/rccl_startup_gfx90a/run_vllm_startup.sh
#
# The layout deliberately matches run_vllm_demo_multinode.sh (one task per node, vLLM
# spawning its own workers) rather than the guide's 8-tasks-per-node layout, because the
# point is to measure OUR documented multi-node baseline. That is also why a per-task
# --cpu-bind mask is absent here: with one task per node it cannot express per-GCD
# affinity. The binding question is tested properly in run_rccl_probe.sh instead.
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
MODE="${MODE:-probe}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
LOAD_FORMAT_DEFAULT="${LOAD_FORMAT}"
VARIANTS_ONLY="${VARIANTS_ONLY:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching}"

# name | extra env (space separated KEY=VAL, or -)
VARIANTS=(
  "baseline|-"
  "socket_ifname|NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3"
  "guide_pair|NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB"
  "runtime_connect_off|NCCL_RUNTIME_CONNECT=0"
  "nchannels_8|NCCL_MAX_NCHANNELS=8"
  "streamer|LOAD_FORMAT=runai_streamer RUNAI_STREAMER_CONCURRENCY=4"
)
RUNG3="baseline socket_ifname nchannels_8"

case "${MODE}" in
  probe) SELECTED="${RUNG3}" ;;
  sweep) SELECTED="" ;;
  *) echo "ERROR: unknown MODE=${MODE} (expected probe or sweep)" >&2; exit 2 ;;
esac
[ -n "${VARIANTS_ONLY}" ] && SELECTED="${VARIANTS_ONLY}"

NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE}}"
PP_SIZE="${PP_SIZE:-${NNODES}}"

echo "Model: ${MODEL}"
echo "Mode: ${MODE}  nodes=${NNODES}  TP=${TP_SIZE}  PP=${PP_SIZE}"
echo "Container: ${CONTAINER}"
echo "Startup timeout: ${STARTUP_TIMEOUT_S}s per variant"

# A stalled LAIFS mount is indistinguishable from the startup stall under
# investigation, so refuse to start rather than mismeasure. run_vllm_demo_multinode.sh
# has no such guard, which is one reason its slow startups were never explained.
LAIFS_PROBE_TIMEOUT_S="${LAIFS_PROBE_TIMEOUT_S:-300}"
for path in /appl/local/laifs/modules "$(dirname "${CONTAINER}")"; do
  echo "probing ${path} (up to ${LAIFS_PROBE_TIMEOUT_S}s)..."
  if ! timeout "${LAIFS_PROBE_TIMEOUT_S}" ls "${path}" >/dev/null 2>&1; then
    echo "ERROR: ${path} is unreachable (LAIFS mount stalled). Aborting." >&2
    exit 1
  fi
done

module load Local-LAIF lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_runtime/${SLURM_JOB_ID}"
RESULTS_HOST="${WORKDIR}/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${RESULTS_HOST}"
echo "Results: ${RESULTS_HOST}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

# Persistent per-user MIOpen cache, as the guide prescribes. The repo's own launchers
# use a per-job mktemp instead, which re-pays kernel compilation on every launch; that
# is a startup cost worth separating from the stall.
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun --ntasks="${NNODES}" --ntasks-per-node=1 \
  mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"
export MODEL PORT TP_SIZE PP_SIZE DISTRIBUTED_EXECUTOR_BACKEND EXTRA_VLLM_ARGS NNODES
export RESULTS_DIR="/work/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"

srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  bash /work/repro/rccl_startup_gfx90a/collect_env.sh || true

for row in "${VARIANTS[@]}"; do
  name="${row%%|*}"
  var_env="${row#*|}"

  if [ -n "${SELECTED}" ] && ! grep -qw "${name}" <<< "${SELECTED}"; then
    continue
  fi

  # Refuse to start a variant that cannot finish: a row cut off by the wall clock looks
  # identical to a stall, and that ambiguity is what this whole case is trying to remove.
  if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    remaining=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    needed=$(( STARTUP_TIMEOUT_S + 300 ))
    if [ "${remaining}" -lt "${needed}" ]; then
      echo "SKIPPING remaining variants: ${remaining}s of walltime left, need ${needed}s."
      echo "Resubmit with VARIANTS_ONLY to finish the table."
      break
    fi
  fi

  echo
  echo "============ variant ${name} ============"

  # Fresh port per variant so a torn-down rendezvous cannot be reused by the next one.
  export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"
  # Per-variant log directory, so timelines cannot be attributed to the wrong variant.
  VARIANT_LOGS="${RUNTIME_DIR}/${name}"
  mkdir -p "${VARIANT_LOGS}"
  export LOG_SUBDIR="${name}"

  unset NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_RUNTIME_CONNECT NCCL_MAX_NCHANNELS \
        NCCL_NCHANNELS_PER_NET_PEER NCCL_PROTO NCCL_NET \
        FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RX_MATCH_MODE
  # Back to the job-level default every iteration, so a variant that sets these cannot
  # leak into the ones after it regardless of the order rows are selected in.
  export LOAD_FORMAT="${LOAD_FORMAT_DEFAULT}"
  export RUNAI_STREAMER_CONCURRENCY=1
  if [ "${var_env}" != "-" ] && [ -n "${var_env}" ]; then
    for kv in ${var_env}; do export "${kv?}"; done
  fi
  export NCCL_DEBUG=WARN
  echo "env: ${var_env}"
  echo "load format: ${LOAD_FORMAT:-<vllm default>}"

  srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL \
    singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    bash /work/repro/rccl_startup_gfx90a/in_container_vllm.sh &
  LAUNCH_PID=$!

  # This is the number run_vllm_demo_multinode.sh computes into $SECONDS and then
  # discards. Recording it is the whole point of this rung.
  READY_URL="http://127.0.0.1:${PORT}/v1/models"
  SECONDS=0
  VERDICT="STALL"
  while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      VERDICT="DIED"
      break
    fi
    if curl -fsS --max-time 5 "${READY_URL}" >/dev/null 2>&1; then
      VERDICT="READY"
      break
    fi
    sleep "${STARTUP_POLL_S}"
  done
  STARTUP_SECONDS="${SECONDS}"
  echo "variant ${name}: verdict=${VERDICT} startup_seconds=${STARTUP_SECONDS}"

  # Tear the server down before the next variant. Killing the srun is not enough: the
  # vLLM workers it spawned keep running and keep their HBM, so the next variant would
  # either OOM or measure a machine that is still busy. Reap them on every node and
  # wait for the memory to actually come back.
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
  srun --overlap --ntasks="${NNODES}" --ntasks-per-node=1 \
    bash -c 'pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::" 2>/dev/null; exit 0' || true
  sleep 20
  # Confirm the GPUs are actually idle before trusting the next measurement.
  srun --overlap --ntasks=1 --ntasks-per-node=1 \
    bash -c 'rocm-smi --showmemuse 2>/dev/null | grep -i "used memory" | head -8 || true' || true

  srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    python3 /work/repro/rccl_startup_gfx90a/phase_timeline.py \
    --logs "/runtime/${name}/vllm_server_rank*.log" \
    --variant "${name}" \
    --results-dir "${RESULTS_DIR}" \
    --startup-seconds "${STARTUP_SECONDS}" \
    --verdict "${VERDICT}" || true
done

echo
echo "=== all variants done; timelines in ${RESULTS_HOST} ==="
grep -H '"startup_seconds"\|"verdict"' "${RESULTS_HOST}"/timeline_*.json 2>/dev/null || true
