#!/bin/bash
# Inside the container on one GCD (via dev.sh): build the tiny config, serve it with dummy
# weights through the plugin, run smoke.py. EXTRA="..." appends vllm serve flags.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SNAP="$(ls -d /scratch/project_462000131/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/* | head -n1)"
TINY="/scratch/project_462000131/${USER}/dsv41-tiny"
LOG="${HERE}/../../benchmarks/results/dsv41-tiny"; mkdir -p "${LOG}"
LOG="${LOG}/$(date +%m%d-%H%M%S).log"
python "${HERE}/make_tiny_config.py" "${SNAP}" "${TINY}"
export VLLM_USE_V2_MODEL_RUNNER=1 AMDGCN_USE_BUFFER_OPS="${AMDGCN_USE_BUFFER_OPS:-0}"
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT="/scratch/project_462000131/${USER}/vllm-cache"
PORT=$((30000 + RANDOM % 20000))
vllm serve "${TINY}" --served-model-name tiny --port "${PORT}" --load-format dummy \
  --tensor-parallel-size "${TP:-1}" --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching --kv-cache-dtype fp8 \
  --moe-backend triton_unfused ${EAGER---enforce-eager} ${EXTRA:-} > "${LOG}" 2>&1 &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null; wait ${SERVER} 2>/dev/null' EXIT
SECONDS=0
until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER}" 2>/dev/null || [ "${SECONDS}" -gt 1200 ]; then
    echo "SERVER_FAILED after ${SECONDS}s (${LOG})"
    grep -vE "amdsmi|pynvml|FutureWarning" "${LOG}" | grep -E "Error|error|Traceback|File \"|raise|^\S" | tail -40
    exit 1
  fi
  sleep 3
done
echo "SERVER_READY after ${SECONDS}s (${LOG})"
python "${HERE}/smoke.py" "http://127.0.0.1:${PORT}" tiny; STATUS=$?
[ "${STATUS}" != 0 ] && grep -vE "amdsmi|pynvml" "${LOG}" | grep -E "Error|Traceback|File \"" | tail -30
exit "${STATUS}"
