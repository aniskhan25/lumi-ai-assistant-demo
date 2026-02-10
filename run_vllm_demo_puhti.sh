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

export MODEL PORT MAX_MODEL_LEN

apptainer exec --nv --bind "${WORKDIR}:/work" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export MODEL="${MODEL}"

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
