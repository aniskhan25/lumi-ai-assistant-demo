#!/bin/bash
#SBATCH --job-name=lumi-vllm-demo
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=demo-%j.out
#SBATCH --error=demo-%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
TP_SIZE="${TP_SIZE:-1}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/project_462000131/${USER}/vllm_runtime/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

echo "Model: ${MODEL}"
echo "Port: ${PORT}  TP_SIZE: ${TP_SIZE}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
MIOPEN_DIR="$(mktemp -d)"

export MODEL PORT TP_SIZE STARTUP_TIMEOUT_S STARTUP_POLL_S EXTRA_VLLM_ARGS MIOPEN_DIR

singularity run "${BIND_ARGS[@]}" "${CONTAINER}" bash /work/launch_vllm_single.sh &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

LOG_PATH="${RUNTIME_DIR}/vllm_server.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
SECONDS=0
while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
  kill -0 "${VLLM_PID}" 2>/dev/null || break

  if curl -fsS --max-time 5 "${READY_URL}" >/dev/null 2>&1; then
    echo "vLLM ready."
    echo "vLLM server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID:-unknown})."
    echo "Server log: ${LOG_PATH}"
    echo "Run queries/benchmarks from another shell via: srun --jobid <jobid> --overlap ..."
    wait "${VLLM_PID}"
    exit 0
  fi

  sleep "${STARTUP_POLL_S}"
done

echo "vLLM failed to start. Last server log lines:" >&2
tail -n 80 "${LOG_PATH}" >&2 || true
exit 1
