#!/bin/bash
#SBATCH --job-name=puhti-vllm-demo
#SBATCH --account=project_2014553
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --output=demo-%j.out
#SBATCH --error=demo-%j.err

set -euo pipefail

CONTAINER="/appl/soft/ai/wrap/pytorch-2.9/container.sif"
MODEL="/scratch/project_2014553/anisrahm/models/Mistral-7B-Instruct-v0.2"
PORT="8000"
MAX_MODEL_LEN="4096"

WORKDIR="$(pwd)"

BIND_ARGS=(--bind "${WORKDIR}:/work")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

export MODEL PORT MAX_MODEL_LEN

apptainer exec --nv "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export MODEL PORT MAX_MODEL_LEN
export HF_HOME="/work/.hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  > /work/vllm_server.log 2>&1 &

VLLM_PID=$!
cleanup() {
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

python - <<'PY'
import json
import os
import time
import urllib.request

port = int(os.environ["PORT"])
base_url = f"http://127.0.0.1:{port}/v1/models"

for attempt in range(60):
    try:
        with urllib.request.urlopen(base_url, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                ids = [m.get('id') for m in data.get('data', [])]
                print(f"vLLM ready. Models: {ids}")
                break
    except Exception as e:
        if attempt == 59:
            raise SystemExit(f"vLLM did not become ready: {e}")
        time.sleep(2)
PY

python /work/demo_agent.py --base-url "http://127.0.0.1:${PORT}/v1"
EOS
