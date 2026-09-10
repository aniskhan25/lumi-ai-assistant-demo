#!/bin/bash
set -euo pipefail

export NODE_RANK="${SLURM_PROCID}"
LOG_PATH="/runtime/vllm_server_rank${NODE_RANK}.log"
exec > "${LOG_PATH}" 2>&1

echo "=== launcher rank ${NODE_RANK} on host $(hostname) ==="

cd /work

export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
# A per-job mktemp here re-pays MIOpen kernel compilation on every launch: measured at
# ~167 s of a cold 8-node vLLM startup (290 s cold vs 123 s warm). Pin per user instead,
# as the LUMI AI Guide does. /tmp is node-local, so this only helps when Slurm reuses
# nodes -- which is still most of the benefit for repeated runs.
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-/tmp/miopen-cache-${USER}}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/miopen-config-${USER}}"
export MIOPEN_USER_DB="${MIOPEN_USER_DB_PATH}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

# Multi-node RCCL hangs indefinitely creating/using additional communicators with the
# vendor-default MR cache monitor (memhooks), which does not see ROCm memory remapping.
# 0 hangs in 18 attempts with this set, against 21/23 without, and it costs no collective
# bandwidth. See repro/rccl_startup_gfx90a/FINDINGS.md and
# github.com/lumi-ai-factory/laifs-container-recipes/issues/44
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"

# Cap RunAI streamer RAM buffer to prevent OOM on large checkpoints.
# Without a limit the streamer accumulates all loaded tensors in RAM before
# transferring to GPU; for checkpoints exceeding per-node RAM this fills
# memory and triggers the OOM killer.
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
if [ "${NODE_RANK}" != "0" ]; then
  VLLM_CMD+=(--headless)
fi
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_VLLM_ARGS}"
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
