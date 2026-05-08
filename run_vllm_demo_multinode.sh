#!/bin/bash
#SBATCH --job-name=lumi-vllm-mn-demo
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=demo-mn-%j.out
#SBATCH --error=demo-mn-%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif}"
MODEL="${MODEL:-/scratch/project_462000131/anisrahm/models/Mistral-7B-Instruct-v0.2}"
PORT="${PORT:-8000}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
ROCM_COMPAT_MODE="${ROCM_COMPAT_MODE:-1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 10000)))}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

ensure_host_python() {
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  # Try to initialize module support in non-interactive Slurm shells.
  if ! type module >/dev/null 2>&1; then
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh || true
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash || true
  fi

  # LUMI module path + pytorch module provide host-side python.
  if type module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles/ >/dev/null 2>&1 || true
    module load pytorch >/dev/null 2>&1 || true
  fi

  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  return 1
}

HOST_PYTHON="$(ensure_host_python)" || {
  echo "No host python found. Tried automatic module load (module use/load pytorch)." >&2
  echo "Load module python manually and resubmit." >&2
  exit 127
}
echo "Using host python for readiness checks: ${HOST_PYTHON}"

NNODES="${SLURM_NNODES:-1}"
if [ "${NNODES}" -lt 2 ]; then
  echo "This launcher is for multi-node runs. Set #SBATCH --nodes to 2+." >&2
  exit 2
fi

GPUS_PER_NODE_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
if [[ "${GPUS_PER_NODE_RAW}" =~ ^([0-9]+) ]]; then
  GPUS_PER_NODE="${BASH_REMATCH[1]}"
else
  GPUS_PER_NODE="1"
fi

# Common vLLM multi-node setting:
# TP = GPUs per node, PP = number of nodes.
TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"
PP_SIZE="${PP_SIZE:-${NNODES}}"

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_BASE="/scratch/project_462000131/${USER}/vllm_runtime"
RUNTIME_DIR="${RUNTIME_BASE}/${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
if [ -z "${HEAD_NODE}" ]; then
  echo "Could not resolve head node from SLURM_JOB_NODELIST." >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"
export CONTAINER MODEL PORT TP_SIZE PP_SIZE VLLM_USE_V1 ENFORCE_EAGER STARTUP_TIMEOUT_S STARTUP_POLL_S ROCM_COMPAT_MODE
export DTYPE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
export NNODES MASTER_ADDR MASTER_PORT WORKDIR RUNTIME_DIR HEAD_NODE

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
else
  echo "No container runtime found (expected apptainer or singularity)." >&2
  echo "Load the container runtime module first, then resubmit." >&2
  exit 127
fi
export CONTAINER_RUNTIME

echo "Launching multi-node vLLM:"
echo "  nodes=${NNODES}, gpus_per_node=${GPUS_PER_NODE}, TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}"
echo "  head node=${HEAD_NODE}, master addr=${MASTER_ADDR}, master port=${MASTER_PORT}"

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --export=ALL \
  bash -lc '
set -euo pipefail
NODE_RANK="${SLURM_PROCID}"
LAUNCH_LOG="${RUNTIME_DIR}/launcher_rank${NODE_RANK}.log"
exec > "${LAUNCH_LOG}" 2>&1
echo "=== launcher rank ${NODE_RANK} on host $(hostname) ==="
echo "WORKDIR=${WORKDIR}"
echo "RUNTIME_DIR=${RUNTIME_DIR}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")
if [ -d "${MODEL}" ]; then
  BIND_ARGS+=(--bind "${MODEL}:${MODEL}")
fi

"${CONTAINER_RUNTIME}" exec --rocm "${BIND_ARGS[@]}" "${CONTAINER}" bash -s <<'"'"'EOS'"'"'
set -euo pipefail
cd /work

NODE_RANK="${SLURM_PROCID}"
echo "=== container rank ${NODE_RANK} on host $(hostname) ==="
export HOME="/runtime"
export XDG_CACHE_HOME="/runtime/.cache"
export HF_HOME="/runtime/.cache/huggingface"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}"
LOG_PATH="/runtime/vllm_server_rank${NODE_RANK}.log"

# Ray on ROCm expects HIP_VISIBLE_DEVICES and may fail if only
# ROCR_VISIBLE_DEVICES is set by the launcher environment.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1
if [ "${ROCM_COMPAT_MODE}" = "1" ]; then
  # More stable on some ROCm stacks for TP/PP runs.
  export TORCH_COMPILE_DISABLE=1
  export VLLM_USE_TRITON_FLASH_ATTN=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

VLLM_CMD=(
  vllm serve "${MODEL}"
  --host 127.0.0.1
  --port "${PORT}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
)
if [ "${NODE_RANK}" != "0" ]; then
  VLLM_CMD+=(--headless)
fi
if [ "${ENFORCE_EAGER}" = "1" ]; then
  VLLM_CMD+=(--enforce-eager)
fi
if [ -n "${GPU_MEMORY_UTILIZATION}" ]; then
  VLLM_CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [ -n "${MAX_MODEL_LEN}" ]; then
  VLLM_CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
  VLLM_CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [ -n "${MAX_NUM_SEQS}" ]; then
  VLLM_CMD+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

echo "Starting: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}" > "${LOG_PATH}" 2>&1
EOS
' > "${RUNTIME_DIR}/srun_step.log" 2>&1 &

LAUNCH_PID=$!
export LAUNCH_PID
cleanup() {
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! "${HOST_PYTHON}" - <<'PY'
import os
import sys
import time
import urllib.request

head = os.environ["HEAD_NODE"]
port = int(os.environ["PORT"])
launch_pid = int(os.environ["LAUNCH_PID"])
timeout_s = int(os.environ.get("STARTUP_TIMEOUT_S", "1800"))
poll_s = float(os.environ.get("STARTUP_POLL_S", "2"))
deadline = time.time() + timeout_s

# Prefer hostname, fallback to loopback when batch script runs on head node.
urls = [f"http://{head}:{port}/v1/models", f"http://127.0.0.1:{port}/v1/models"]

while time.time() < deadline:
    try:
        os.kill(launch_pid, 0)
    except OSError:
        raise SystemExit("vLLM launcher step exited before readiness check passed.")

    for url in urls:
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
  echo "vLLM multi-node startup failed. Tail logs:" >&2
  echo "--- ${RUNTIME_DIR}/srun_step.log ---" >&2
  tail -n 80 "${RUNTIME_DIR}/srun_step.log" >&2 || true
  echo "--- ${RUNTIME_DIR} (ls -lah) ---" >&2
  ls -lah "${RUNTIME_DIR}" >&2 || true
  for rank in $(seq 0 $((NNODES - 1))); do
    echo "--- ${RUNTIME_DIR}/launcher_rank${rank}.log ---" >&2
    tail -n 80 "${RUNTIME_DIR}/launcher_rank${rank}.log" >&2 || true
    echo "--- ${RUNTIME_DIR}/vllm_server_rank${rank}.log ---" >&2
    tail -n 80 "${RUNTIME_DIR}/vllm_server_rank${rank}.log" >&2 || true
  done
  exit 1
fi

echo "vLLM multi-node server is ready at http://127.0.0.1:${PORT}/v1 (job ${SLURM_JOB_ID})."
echo "Head node: ${HEAD_NODE}"
echo "Run queries from another shell pinned to head node:"
echo "  srun --jobid ${SLURM_JOB_ID} --overlap -w ${HEAD_NODE} --export=ALL python /scratch/project_462000131/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:${PORT}/v1 --question \"test\""
echo "Logs:"
echo "  ${RUNTIME_DIR}/srun_step.log"
echo "  ${RUNTIME_DIR}/launcher_rank0.log"
echo "  ${RUNTIME_DIR}/launcher_rank1.log ... (one per node rank)"
echo "  ${RUNTIME_DIR}/vllm_server_rank0.log"
echo "  ${RUNTIME_DIR}/vllm_server_rank1.log ... (one per node rank)"
echo "Keeping this job alive until the multi-node vLLM step exits."

wait "${LAUNCH_PID}"
