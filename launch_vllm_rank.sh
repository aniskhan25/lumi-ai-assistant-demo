#!/bin/bash
set -euo pipefail

MODEL_NAME="$1"
shift

export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

MULTINODE_ARGS=""
if [[ "${SLURM_NNODES}" -gt 1 ]]; then
  MULTINODE_ARGS="--distributed-executor-backend mp --nnodes ${SLURM_NNODES} --node-rank ${SLURM_PROCID} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}"
  if [[ "${SLURM_PROCID}" -ne 0 ]]; then
    MULTINODE_ARGS="${MULTINODE_ARGS} --headless"
  fi
fi

CMD="vllm serve ${MODEL_NAME} ${MULTINODE_ARGS} $*"
echo "${CMD}"
exec ${CMD}
