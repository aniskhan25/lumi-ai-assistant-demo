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
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTILIZATION="0.85"
SWAP_SPACE_GB="0"
MAX_NUM_SEQS="1"
ATTN_BACKEND="TORCH_SDPA"

WORKDIR="$(pwd)"
RUNTIME_BASE="/scratch/project_2014553/anisrahm/vllm_runtime"
RUNTIME_DIR="${RUNTIME_BASE}/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
else
  echo "Model directory does not exist on host: ${MODEL}" >&2
  exit 2
fi

export MODEL PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION SWAP_SPACE_GB MAX_NUM_SEQS ATTN_BACKEND

apptainer exec --nv "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'EOS'
set -euo pipefail
cd /work
export MODEL PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION SWAP_SPACE_GB MAX_NUM_SEQS ATTN_BACKEND
export HF_HUB_DISABLE_TELEMETRY=1
export HOME="/runtime"
export HF_HOME="/runtime/.hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="/runtime/.cache"
export FLASHINFER_WORKSPACE_DIR="/runtime/.flashinfer"
export TRITON_CACHE_DIR="/runtime/.triton"
export VLLM_ATTENTION_BACKEND="${ATTN_BACKEND}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_DIR}" "${TRITON_CACHE_DIR}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --swap-space "${SWAP_SPACE_GB}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  > /work/vllm_server.log 2>&1 &

VLLM_PID=$!
export VLLM_PID
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
pid = int(os.environ["VLLM_PID"])
log_path = "/work/vllm_server.log"
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
        try:
            os.kill(pid, 0)
        except OSError:
            print("vLLM process exited before readiness. Last server log lines:")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                for line in lines[-80:]:
                    print(line.rstrip())
            raise SystemExit(f"vLLM failed before becoming ready: {e}")
        if attempt == 59:
            print("vLLM readiness timeout. Last server log lines:")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                for line in lines[-80:]:
                    print(line.rstrip())
            raise SystemExit(f"vLLM did not become ready: {e}")
        time.sleep(2)
PY

python /work/demo_agent.py --base-url "http://127.0.0.1:${PORT}/v1"
EOS
