#!/bin/bash
#SBATCH --job-name=puhti-vllm-demo
#SBATCH --account=project_2014553
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=demo-%j.out
#SBATCH --error=demo-%j.err

set -euo pipefail

CONTAINER="/appl/soft/ai/wrap/pytorch-2.9/container.sif"
MODEL="/scratch/project_2014553/anisrahm/models/Mistral-7B-Instruct-v0.2"
PORT="8000"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_BASE="/scratch/project_2014553/anisrahm/vllm_runtime"
RUNTIME_DIR="${RUNTIME_BASE}/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

export MODEL PORT

apptainer exec --nv "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export HOME="/runtime"
export XDG_CACHE_HOME="/runtime/.cache"
export HF_HOME="/runtime/.cache/huggingface"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}"
LOG_PATH="/runtime/vllm_server.log"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
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

python /work/demo_agent.py --base-url "http://127.0.0.1:${PORT}/v1"
EOS
