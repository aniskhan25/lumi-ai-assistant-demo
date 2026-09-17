#!/bin/bash
# Per-task driver, run inside the container by run_block.sh. One rank, one slot.
#
# This exists because the inline `srun ... bash -c "..."` it replaces needed three
# levels of quote escaping to set a per-rank variable, and got it wrong: NCCL_DEBUG_FILE
# substitutes only %h (hostname) and %p (pid), never %d for rank, so `nccl_x_rank%d.log`
# was written literally and nothing could ever find rank 0's log. The bug study's
# in_container_probe.sh solved the same problem the same way.
#
# Expects from the environment: SLOT_NAME, RESULTS_DIR, REL_DIR, PROFILE_ARGS,
# and SLOT_DEBUG (0/1).

set -uo pipefail

export RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID}"
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HOME=/runtime

# Node-local, and named by rank rather than by a format specifier NCCL does not
# support. Lustre-resident INFO logs from every rank cost job 22115267 its whole
# walltime, so this stays on /tmp and only rank 0's copy is kept.
if [ "${SLOT_DEBUG:-0}" = "1" ]; then
  export NCCL_DEBUG_FILE="/tmp/nccl_${SLOT_NAME}_rank${SLURM_PROCID}.log"
fi

python3 "/work/${REL_DIR}/collective_profile.py" \
  --variant "${SLOT_NAME}" --results-dir "${RESULTS_DIR}" ${PROFILE_ARGS:-}
rc=$?

# One rank's INIT/TUNING trace is enough: every rank logs the same configuration, and
# 16 copies of it on Lustre is what broke the first attempt.
if [ "${SLOT_DEBUG:-0}" = "1" ] && [ "${SLURM_PROCID}" = "0" ]; then
  src="/tmp/nccl_${SLOT_NAME}_rank0.log"
  [ -f "${src}" ] && cp "${src}" "${RESULTS_DIR}/debug_${SLOT_NAME}_rank0.log" 2>/dev/null
  # If NCCL wrote somewhere unexpected, say so in-band rather than leaving an empty
  # Layer 2 that looks like "the knob did nothing".
  [ -f "${src}" ] || echo "WARN: expected ${src}, found: $(ls /tmp/nccl_* 2>/dev/null | head -3)" >&2
fi

exit ${rc}
