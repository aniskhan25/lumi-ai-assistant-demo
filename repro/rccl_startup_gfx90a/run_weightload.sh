#!/bin/bash
#SBATCH --job-name=rccl-weightload
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --output=rccl-weightload-%j.out
#SBATCH --error=rccl-weightload-%j.err

# Decide whether the reporter's second symptom -- the weight streamer sitting at
# "0% Completed" for 850+ s -- has anything to do with RCCL at all.
#
# One node, so there is no inter-node communication to blame. If the stall reproduces
# here it is the loader or the filesystem, not the fabric, and it needs a different
# answer. This repo has been wrong on this axis before: FINDINGS.md in the sibling case
# carries a commit "Retract runai_streamer claim", so the two symptoms get separated by
# measurement rather than by assumption.
#
#   sbatch repro/rccl_startup_gfx90a/run_weightload.sh                    # gpt-oss-120b
#   READ_ONLY=1 MODEL=moonshotai/Kimi-K2-Instruct-0905 sbatch ...         # Lustre read only
#   MODEL=... sbatch ...
#
# READ_ONLY=1 skips vLLM and only measures raw checkpoint read throughput, which is the
# only way to characterise the 959 GB Kimi-K2 checkpoint on a single node: it does not
# fit in one node's HBM, but its read cost is measurable regardless.
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
READ_ONLY="${READ_ONLY:-0}"
READ_BUDGET_GB="${READ_BUDGET_GB:-64}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.90}"

# name | LOAD_FORMAT | RUNAI_STREAMER_CONCURRENCY
# dummy is the no-I/O control: whatever time remains is not the filesystem.
VARIANTS=(
  "dummy|dummy|1"
  "default_loader||1"
  "streamer_c1|runai_streamer|1"
  "streamer_c4|runai_streamer|4"
  "streamer_c16|runai_streamer|16"
)

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
echo "Model: ${MODEL}"
echo "Results: ${RESULTS_HOST}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

export RESULTS_DIR="/work/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"
export MODEL PORT TP_SIZE EXTRA_VLLM_ARGS
# Single node: vLLM needs no rendezvous, which is the point.
export PP_SIZE=1 NNODES=1 MASTER_ADDR=127.0.0.1
export DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"

# --- step 1: raw checkpoint read throughput --------------------------------
# Weights live on /scratch (Lustre), never /flash (NVMe), in every launcher in this
# repo. If Lustre alone cannot feed the loader fast enough, no loader setting will fix
# the reporter's second symptom.
echo
echo "=== step 1: raw read throughput from Lustre (budget ${READ_BUDGET_GB} GB) ==="
srun --ntasks=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  bash /work/repro/rccl_startup_gfx90a/lustre_read.sh "${MODEL}" "${READ_BUDGET_GB}" || true

if [ "${READ_ONLY}" = "1" ]; then
  echo "READ_ONLY=1, stopping after the read measurement."
  exit 0
fi

srun --ntasks=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  bash /work/repro/rccl_startup_gfx90a/collect_env.sh || true

# --- step 2: single-node load-format sweep --------------------------------
for row in "${VARIANTS[@]}"; do
  name="${row%%|*}"
  rest="${row#*|}"
  fmt="${rest%%|*}"
  conc="${rest#*|}"

  echo
  echo "============ variant ${name} (load-format=${fmt:-<vllm default>} concurrency=${conc}) ============"

  VARIANT_LOGS="${RUNTIME_DIR}/${name}"
  mkdir -p "${VARIANT_LOGS}"
  export LOG_SUBDIR="${name}"
  export LOAD_FORMAT="${fmt}"
  export RUNAI_STREAMER_CONCURRENCY="${conc}"
  export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"

  srun --ntasks=1 --kill-on-bad-exit=0 --export=ALL \
    singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    bash /work/repro/rccl_startup_gfx90a/in_container_vllm.sh &
  LAUNCH_PID=$!

  READY_URL="http://127.0.0.1:${PORT}/v1/models"
  SECONDS=0
  VERDICT="STALL"
  while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then VERDICT="DIED"; break; fi
    if curl -fsS --max-time 5 "${READY_URL}" >/dev/null 2>&1; then VERDICT="READY"; break; fi
    sleep "${STARTUP_POLL_S}"
  done
  STARTUP_SECONDS="${SECONDS}"
  echo "variant ${name}: verdict=${VERDICT} startup_seconds=${STARTUP_SECONDS}"

  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
  sleep 15

  srun --ntasks=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    python3 /work/repro/rccl_startup_gfx90a/phase_timeline.py \
    --logs "/runtime/${name}/vllm_server_rank*.log" \
    --variant "${name}" --results-dir "${RESULTS_DIR}" \
    --startup-seconds "${STARTUP_SECONDS}" --verdict "${VERDICT}" || true
done

echo
echo "=== done; the weights_start -> weights_done gap in each timeline is the answer ==="
