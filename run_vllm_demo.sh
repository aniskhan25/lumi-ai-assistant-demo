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
MISTRAL_MODEL_DEFAULT="${MISTRAL_MODEL_DEFAULT:-/scratch/project_462000131/anisrahm/models/Mistral-7B-Instruct-v0.2}"
QWEN_MODEL_DEFAULT="${QWEN_MODEL_DEFAULT:-/scratch/project_462000131/anisrahm/models/Qwen/Qwen3.6-35B-A3B}"
MODEL="${MODEL:-}"
PORT="${PORT:-8000}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/project_462000131/${USER}/vllm_runtime/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

GPU_COUNT="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
if [[ "${GPU_COUNT}" =~ ^([0-9]+) ]]; then
  GPU_COUNT="${BASH_REMATCH[1]}"
else
  GPU_COUNT="1"
fi
TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"

if [ -z "${MODEL}" ]; then
  if [ "${GPU_COUNT}" -gt 1 ]; then
    MODEL="${QWEN_MODEL_DEFAULT}"
  else
    MODEL="${MISTRAL_MODEL_DEFAULT}"
  fi
fi

if [ "${TP_SIZE}" -gt "${GPU_COUNT}" ]; then
  echo "TP_SIZE=${TP_SIZE} exceeds allocated GPUs (${GPU_COUNT})." >&2
  exit 2
fi

echo "Model: ${MODEL}"
echo "Port: ${PORT}  TP_SIZE: ${TP_SIZE}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi
MIOPEN_DIR="$(mktemp -d)"

export MODEL PORT TP_SIZE STARTUP_TIMEOUT_S STARTUP_POLL_S EXTRA_VLLM_ARGS MIOPEN_DIR

singularity run "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail

cd /work
export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}/cache"
export MIOPEN_USER_DB="${MIOPEN_DIR}/config"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB}"
LOG_PATH="/runtime/vllm_server.log"

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

VLLM_CMD=(
  vllm serve "${MODEL}"
  --host 127.0.0.1
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --load-format runai_streamer
)
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_VLLM_ARGS}"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

"${VLLM_CMD[@]}" > "${LOG_PATH}" 2>&1 &
VLLM_PID=$!
export VLLM_PID

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! python - <<'PY'
import os
import sys
import time
import urllib.request

port = int(os.environ["PORT"])
pid = int(os.environ["VLLM_PID"])
timeout_s = int(os.environ.get("STARTUP_TIMEOUT_S", "900"))
poll_s = float(os.environ.get("STARTUP_POLL_S", "2"))
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + timeout_s

while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("vLLM process exited before readiness check passed.")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("vLLM ready.")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(poll_s)

raise SystemExit(f"vLLM did not become ready in time (timeout={timeout_s}s).")
PY
then
  echo "vLLM failed to start. Last server log lines:" >&2
  tail -n 80 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "vLLM server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID:-unknown})."
echo "Server log: ${LOG_PATH}"
echo "Run queries/benchmarks from another shell via: srun --jobid <jobid> --overlap ..."
wait "${VLLM_PID}"
EOS
