#!/bin/bash
# In-container driver for one probe variant. Runs once per task (one per GCD).
set -uo pipefail

VARIANT="${1:-baseline}"

cd /work

export HOME="/runtime"
export XDG_CACHE_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" 2>/dev/null || true

# Slurm hands each task its own GCD; mirror the repo's launchers so the probe sees the
# same device visibility our servers do.
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"

# Sample Cassini telemetry once per node, on either side of the run. If the stall is
# CXI resource exhaustion the deltas say so; if it is the bootstrap interface they stay
# flat, which is what distinguishes H-A from H-B.
if [ "${SLURM_LOCALID}" = "0" ]; then
  bash /work/repro/rccl_startup_gfx90a/cxi_counters.sh "${VARIANT}" before
fi

python3 /work/repro/rccl_startup_gfx90a/rccl_probe.py \
  --variant "${VARIANT}" \
  --results-dir "${RESULTS_DIR}" \
  --stall-timeout-s "${STALL_TIMEOUT_S:-300}" \
  --tensor-mib "${TENSOR_MIB:-32}" \
  --many-comms "${MANY_COMMS:-0}"
status=$?

if [ "${SLURM_LOCALID}" = "0" ]; then
  bash /work/repro/rccl_startup_gfx90a/cxi_counters.sh "${VARIANT}" after
fi

exit "${status}"
