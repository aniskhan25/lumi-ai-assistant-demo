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
MASTER_PORT="${MASTER_PORT:-1${SLURM_JOB_ID: -4}}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---enable-expert-parallel --all2all-backend deepep_low_latency}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE}}"
PP_SIZE="${PP_SIZE:-${NNODES}}"

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/project_462000131/${USER}/vllm_runtime/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"

export MODEL PORT TP_SIZE PP_SIZE DISTRIBUTED_EXECUTOR_BACKEND EXTRA_VLLM_ARGS
export NNODES MASTER_ADDR MASTER_PORT

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

echo "Launching multi-node vLLM:"
echo "  model=${MODEL}"
echo "  nodes=${NNODES}, gpus_per_node=${SLURM_GPUS_ON_NODE}, TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}"
echo "  head node=${HEAD_NODE}, master addr=${MASTER_ADDR}, master port=${MASTER_PORT}"
echo "  distributed executor backend=${DISTRIBUTED_EXECUTOR_BACKEND}"
echo "  extra vLLM args=${EXTRA_VLLM_ARGS}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL \
  singularity run "${BIND_ARGS[@]}" "${CONTAINER}" bash /work/launch_vllm_multinode_rank.sh &

LAUNCH_PID=$!
cleanup() {
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY_URL="http://127.0.0.1:${PORT}/v1/models"
SECONDS=0
while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
  kill -0 "${LAUNCH_PID}" 2>/dev/null || break

  if curl -fsS --max-time 5 "${READY_URL}" >/dev/null 2>&1; then
    echo "vLLM ready."
    echo "vLLM multi-node server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID})."
    echo "Head node: ${HEAD_NODE}"
    echo "Logs:"
    echo "  ${RUNTIME_DIR}/vllm_server_rank*.log"
    wait "${LAUNCH_PID}"
    exit 0
  fi

  sleep "${STARTUP_POLL_S}"
done

echo "vLLM multi-node startup failed. Tail logs:" >&2
for rank in $(seq 0 $((NNODES - 1))); do
  echo "--- ${RUNTIME_DIR}/vllm_server_rank${rank}.log ---" >&2
  tail -n 80 "${RUNTIME_DIR}/vllm_server_rank${rank}.log" >&2 || true
done
exit 1
