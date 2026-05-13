#!/bin/bash
#SBATCH --job-name=lumi-vllm-mn-demo
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --output=demo-mn-%j.out
#SBATCH --error=demo-mn-%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 10000)))}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-deepep_low_latency}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

module use /appl/local/csc/modulefiles/
module load pytorch
HOST_PYTHON=python

NNODES="${SLURM_NNODES}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE}}"
PP_SIZE="${PP_SIZE:-${SLURM_NNODES}}"

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/project_462000131/${USER}/vllm_runtime/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
if [ -z "${HEAD_NODE}" ]; then
  echo "Could not resolve head node from SLURM_JOB_NODELIST." >&2
  exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"

export MODEL PORT TP_SIZE PP_SIZE STARTUP_TIMEOUT_S STARTUP_POLL_S DISTRIBUTED_EXECUTOR_BACKEND ALL2ALL_BACKEND ENABLE_EXPERT_PARALLEL EXTRA_VLLM_ARGS
export NNODES MASTER_ADDR MASTER_PORT WORKDIR RUNTIME_DIR HEAD_NODE

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

echo "Launching multi-node vLLM:"
echo "  model=${MODEL}"
echo "  nodes=${NNODES}, gpus_per_node=${SLURM_GPUS_ON_NODE}, TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}"
echo "  head node=${HEAD_NODE}, master addr=${MASTER_ADDR}, master port=${MASTER_PORT}"
echo "  distributed executor backend=${DISTRIBUTED_EXECUTOR_BACKEND}"
echo "  expert parallel=${ENABLE_EXPERT_PARALLEL}, all2all backend=${ALL2ALL_BACKEND}"

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL \
  singularity run "${BIND_ARGS[@]}" "${CONTAINER}" bash /work/launch_vllm_rank.sh &

LAUNCH_PID=$!
export LAUNCH_PID
cleanup() {
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY_URL="http://127.0.0.1:${PORT}/v1/models"
READY=0
SECONDS=0
while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "vLLM launcher step exited before readiness check passed." >&2
    break
  fi
  if "${HOST_PYTHON}" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5).close()' "${READY_URL}" >/dev/null 2>&1; then
    echo "vLLM ready."
    READY=1
    break
  fi
  sleep "${STARTUP_POLL_S}"
done

if [ "${READY}" != "1" ]; then
  echo "vLLM multi-node startup failed. Tail logs:" >&2
  for rank in $(seq 0 $((NNODES - 1))); do
    echo "--- ${RUNTIME_DIR}/vllm_server_rank${rank}.log ---" >&2
    tail -n 80 "${RUNTIME_DIR}/vllm_server_rank${rank}.log" >&2 || true
  done
  exit 1
fi

echo "vLLM multi-node server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID})."
echo "Head node: ${HEAD_NODE}"
echo "Run queries from another shell pinned to head node:"
echo "  srun --jobid ${SLURM_JOB_ID} --overlap --exact -N1 -n1 -w ${HEAD_NODE} --export=ALL python /scratch/project_462000131/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:${PORT}/v1 --question \"test\""
echo "Logs:"
echo "  ${RUNTIME_DIR}/vllm_server_rank*.log"

wait "${LAUNCH_PID}"
