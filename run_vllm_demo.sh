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

CONTAINER="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif"
MODEL="/scratch/project_462000131/anisrahm/models/Mistral-7B-Instruct-v0.2"
PORT="8000"
TP_SIZE="1"

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_BASE="/scratch/project_462000131/${USER}/vllm_runtime"
RUNTIME_DIR="${RUNTIME_BASE}/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

if [ ! -f "${WORKDIR}/demo_agent.py" ]; then
  echo "demo_agent.py not found at ${WORKDIR}/demo_agent.py" >&2
  echo "Submit from repo root or set REPO_DIR to your repo checkout path." >&2
  exit 2
fi
if [ ! -d "${WORKDIR}/lumi_docs" ]; then
  echo "lumi_docs directory not found at ${WORKDIR}/lumi_docs" >&2
  echo "Submit from repo root or set REPO_DIR to your repo checkout path." >&2
  exit 2
fi

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

export MODEL PORT TP_SIZE

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
export VLLM_USE_V1=0

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enforce-eager \
  > "${LOG_PATH}" 2>&1 &

VLLM_PID=$!
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

for attempt in range(60):
    try:
        with urllib.request.urlopen(base_url, timeout=5) as resp:
            if resp.status == 200:
                print("vLLM ready.")
                sys.exit(0)
    except Exception:
        pass
    if attempt == 59:
        raise SystemExit("vLLM did not become ready in time.")
    else:
        time.sleep(2)
PY
then
  echo "vLLM failed to start. Last server log lines:" >&2
  tail -n 80 "${LOG_PATH}" >&2 || true
  exit 1
fi

echo "vLLM server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID:-unknown})."
echo "Run queries/benchmarks from another shell via: srun --jobid <jobid> --overlap ..."
echo "Keeping this job alive until the vLLM server exits."
wait "${VLLM_PID}"
EOS
