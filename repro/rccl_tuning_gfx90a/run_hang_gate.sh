#!/bin/bash
#SBATCH --job-name=rccl-hang-gate
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:45:00
#SBATCH --output=rccl-gate-%A_%a-%j.out
#SBATCH --error=rccl-gate-%A_%a-%j.err

# Stage 4b: the hang regression gate. Mandatory and independent of any speed result.
#
# A tuning config that reintroduces the memhooks-class startup hang is worthless at any
# throughput, so the candidate has to clear the bug study's own reproducer before it can
# be written into env_tuned.sh.
#
#   sbatch --array=1-5 repro/rccl_tuning_gfx90a/run_hang_gate.sh
#
# ONE ATTEMPT PER ALLOCATION, five allocations. Repeats inside a single job are not
# independent: the first run pays one-time costs the later ones do not, and a stalled
# attempt leaves ranks blocked inside RCCL holding their GCDs, which poisons whatever
# runs next. That flaw produced five retracted findings in the bug study, so the gate
# does not repeat it for the sake of a cheaper job.
#
# The instrument is repro/rccl_startup_gfx90a/rccl_probe.py, reused unmodified.

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
VARIANT="${VARIANT:-cand}"
# 8 communicators is the shape that reproduced the hang: vLLM builds TP, PP, world and
# all2all groups and hits each one's first collective at a different startup stage.
MANY_COMMS="${MANY_COMMS:-8}"
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-300}"

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

LAIFS_PROBE_TIMEOUT_S="${LAIFS_PROBE_TIMEOUT_S:-300}"
for path in /appl/local/laifs/modules "$(dirname "${CONTAINER}")"; do
  if ! timeout "${LAIFS_PROBE_TIMEOUT_S}" ls "${path}" >/dev/null 2>&1; then
    echo "ERROR: ${path} is unreachable (LAIFS mount stalled). Aborting." >&2
    exit 1
  fi
done

module load Local-LAIF lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_runtime/${SLURM_JOB_ID}"
REL="repro/rccl_tuning_gfx90a"
RESULTS_HOST="${WORKDIR}/${REL}/results/gate_${SLURM_JOB_ID}"
RESULTS_CONT="/work/${REL}/results/gate_${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${RESULTS_HOST}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"
export WORLD_SIZE="${SLURM_NPROCS}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export RESULTS_DIR="${RESULTS_CONT}"
export STALL_TIMEOUT_S MANY_COMMS
export NCCL_DEBUG=WARN

# The candidate, exactly as env_tuned.sh would ship it at this scale.
export FI_MR_CACHE_MONITOR=userfaultfd
[ "${VARIANT}" = "cand" ] && export NCCL_MIN_NCHANNELS=32

echo "variant=${VARIANT} nodes=${SLURM_JOB_NUM_NODES} world=${SLURM_NPROCS} comms=${MANY_COMMS}"
echo "nodelist=${SLURM_JOB_NODELIST}"
echo "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-<unset>}  FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR}"

export SLOT_NAME="${VARIANT}" REL_DIR="${REL}" SLOT_DEBUG=0
export SLOT_SCRIPT="/work/repro/rccl_startup_gfx90a/rccl_probe.py"
export PROFILE_ARGS="--many-comms ${MANY_COMMS} --stall-timeout-s ${STALL_TIMEOUT_S}"

set +e
srun --kill-on-bad-exit=0 --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" \
  singularity run "${BIND_ARGS[@]}" "${CONTAINER}" "/work/${REL}/in_container_slot.sh"
rc=$?
set -e
echo "probe exit=${rc}"

srun --overlap --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  bash -c 'pkill -f "[r]ccl_probe\.py" || true' >/dev/null 2>&1 || true

echo "=== verdicts (any STALL fails the gate) ==="
grep -ho '"verdict": *"[A-Z]*"' "${RESULTS_HOST}"/*.json 2>/dev/null | sort | uniq -c || true
