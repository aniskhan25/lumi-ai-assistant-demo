#!/bin/bash
# In-container vLLM launcher for one variant. One task per node; vLLM spawns the
# per-GCD workers itself. Mirrors launch_vllm_multinode_rank.sh so that what is being
# measured is the repo's real multi-node path, with per-variant log separation added.
set -euo pipefail

export NODE_RANK="${SLURM_PROCID}"
LOG_DIR="/runtime/${LOG_SUBDIR:-default}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/vllm_server_rank${NODE_RANK}.log"
exec > "${LOG_PATH}" 2>&1

echo "=== launcher rank ${NODE_RANK} on host $(hostname) ==="

cd /work

export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}"

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

# Only meaningful when --load-format runai_streamer is actually passed. The repo's
# multi-node launcher sets these but never passes the flag, so there they are inert --
# which is exactly why the `streamer` variant has to set both together.
export RUNAI_STREAMER_CONCURRENCY="${RUNAI_STREAMER_CONCURRENCY:-1}"
export RUNAI_STREAMER_MEMORY_LIMIT="${RUNAI_STREAMER_MEMORY_LIMIT:-8}"

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
if [ -n "${LOAD_FORMAT:-}" ]; then
  VLLM_CMD+=(--load-format "${LOAD_FORMAT}")
fi
if [ "${NODE_RANK}" != "0" ]; then
  VLLM_CMD+=(--headless)
fi
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_VLLM_ARGS}"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
