#!/bin/bash
set -euo pipefail

export NODE_RANK="${SLURM_PROCID}"
LOG_PATH="/runtime/vllm_server_rank${NODE_RANK}.log"
exec > "${LOG_PATH}" 2>&1

echo "=== launcher rank ${NODE_RANK} on host $(hostname) ==="

cd /work

export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/vllm-cache"
MIOPEN_DIR="$(mktemp -d)"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}/cache"
export MIOPEN_USER_DB="${MIOPEN_DIR}/config"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB}"

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

VLLM_CMD=(
  vllm serve "${MODEL}"
  --host 127.0.0.1
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
)
if [ "${NODE_RANK}" != "0" ]; then
  VLLM_CMD+=(--headless)
fi
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_VLLM_ARGS}"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
