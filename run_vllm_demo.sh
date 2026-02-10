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
MODEL="/path/to/model"
PORT="8000"
MAX_MODEL_LEN="4096"

WORKDIR="$(pwd)"

BIND_ARGS=(--bind "${WORKDIR}:/work")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

apptainer exec --rocm "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export MODEL="${MODEL}"
export HF_HOME="/work/.hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

python -m vllm.entrypoints.openai.api_server \\
  --model "${MODEL}" \\
  --host 127.0.0.1 \\
  --port "${PORT}" \\
  --max-model-len "${MAX_MODEL_LEN}" \\
  > /work/vllm_server.log 2>&1 &

VLLM_PID=\$!
cleanup() {
  kill \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

python - <<'PY'
import json
import time
import urllib.request

base_url = f"http://127.0.0.1:{int('${PORT}')}/v1/models"

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
