#!/bin/bash
#SBATCH --job-name=lumi-vllm-demo
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=demo-%j.out
#SBATCH --error=demo-%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif}"
MODEL="${MODEL:-/scratch/project_462000131/anisrahm/models/Mistral-7B-Instruct-v0.2}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
ROCM_COMPAT_MODE="${ROCM_COMPAT_MODE:-1}"

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_BASE="/scratch/project_462000131/${USER}/vllm_runtime"
RUNTIME_DIR="${RUNTIME_BASE}/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

export MODEL PORT TP_SIZE VLLM_USE_V1 ENFORCE_EAGER STARTUP_TIMEOUT_S STARTUP_POLL_S ROCM_COMPAT_MODE

GPU_COUNT="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [ "${TP_SIZE}" -gt "${GPU_COUNT}" ]; then
  echo "TP_SIZE=${TP_SIZE} exceeds allocated GPUs (${GPU_COUNT})." >&2
  echo "Increase Slurm GPU allocation or lower TP_SIZE." >&2
  exit 2
fi

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
else
  echo "No container runtime found (expected apptainer or singularity)." >&2
  echo "Load the container runtime module first, then resubmit." >&2
  exit 127
fi

"${CONTAINER_RUNTIME}" exec --rocm "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export HOME="/runtime"
export XDG_CACHE_HOME="/runtime/.cache"
export HF_HOME="/runtime/.cache/huggingface"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}"
LOG_PATH="/runtime/vllm_server.log"

# Ray on ROCm expects HIP_VISIBLE_DEVICES and may fail if only
# ROCR_VISIBLE_DEVICES is set by the launcher environment.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1
if [ "${ROCM_COMPAT_MODE}" = "1" ]; then
  # More stable on some ROCm stacks for TP>1 (avoids Triton/Inductor crashes).
  export TORCH_COMPILE_DISABLE=1
  export VLLM_USE_TRITON_FLASH_ATTN=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --host 127.0.0.1
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
)
if [ "${ENFORCE_EAGER}" = "1" ]; then
  VLLM_CMD+=(--enforce-eager)
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
import time
import urllib.request
import sys

port = int(os.environ["PORT"])
base_url = f"http://127.0.0.1:{port}/v1/models"
pid = int(os.environ["VLLM_PID"])
timeout_s = int(os.environ.get("STARTUP_TIMEOUT_S", "900"))
poll_s = float(os.environ.get("STARTUP_POLL_S", "2"))
deadline = time.time() + timeout_s

while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("vLLM process exited before readiness check passed.")
    try:
        with urllib.request.urlopen(base_url, timeout=5) as resp:
            if resp.status == 200:
                print("vLLM ready.")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(poll_s)

raise SystemExit(f"vLLM did not become ready in time (timeout={timeout_s}s).")
PY
then
  echo "Startup settings: TP_SIZE=${TP_SIZE}, STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S}, ROCM_COMPAT_MODE=${ROCM_COMPAT_MODE}" >&2
  echo "Server log path: ${LOG_PATH}" >&2
  echo "vLLM failed to start. Last server log lines:" >&2
  tail -n 80 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "vLLM server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID:-unknown})."
echo "Server log: ${LOG_PATH}"
echo "Run queries/benchmarks from another shell via: srun --jobid <jobid> --overlap ..."
echo "Keeping this job alive until the vLLM server exits."
wait "${VLLM_PID}"
EOS
