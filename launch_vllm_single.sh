#!/bin/bash
set -euo pipefail

LOG_PATH="/runtime/vllm_server.log"
exec > "${LOG_PATH}" 2>&1

cd /work

export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}/cache"
export MIOPEN_USER_DB="${MIOPEN_DIR}/config"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB}"

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

# Serialize RunAI streamer tensor loading to prevent concurrent buffer spikes
# that exhaust memory when TP workers load in parallel (especially for large models).
export RUNAI_STREAMER_CONCURRENCY="${RUNAI_STREAMER_CONCURRENCY:-1}"

LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"

VLLM_CMD=(
  vllm serve "${MODEL}"
  --host 127.0.0.1
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --load-format "${LOAD_FORMAT}"
)
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_VLLM_ARGS}"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
