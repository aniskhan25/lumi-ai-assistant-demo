#!/bin/bash
#SBATCH --job-name=rccl-startup-probe
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --output=rccl-probe-%j.out
#SBATCH --error=rccl-probe-%j.err

# Time RCCL startup phase by phase with no model and no vLLM, to find out whether the
# reported multi-node startup stall is (H-A) RCCL bootstrapping over a non-HSN
# interface because NCCL_SOCKET_IFNAME is unset, or (H-B) RCCL deferring per-peer
# connection setup so every new communicator pays a fresh burst.
#
#   sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh                # rung 1: 3 variants
#   MODE=sweep sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh     # rung 2: full table
#   MODE=sweep sbatch --nodes=4 repro/rccl_startup_gfx90a/run_rccl_probe.sh
#   MODE=sweep sbatch --nodes=8 repro/rccl_startup_gfx90a/run_rccl_probe.sh
#   MODE=debug sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh     # NCCL INFO logs
#   VARIANTS_ONLY="baseline socket_ifname" sbatch ...                 # pick rows
#
# The task layout (8 tasks/node, 7 cpus/task, --mem-per-gpu=60G) and the CPU bind mask
# below are the LUMI AI Guide's, so the probe measures the configuration the guide
# actually recommends -- and so the cpu_bind variant is a clean A/B, which it cannot be
# in the vLLM path (one task per node there, so a per-task mask says nothing).
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
MODE="${MODE:-probe}"
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-300}"
TENSOR_MIB="${TENSOR_MIB:-32}"
VARIANTS_ONLY="${VARIANTS_ONLY:-}"

# See https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# name | extra env (space separated KEY=VAL, or -) | extra srun flags
# One variable moves per row, ordered by prior probability of being the cause.
VARIANTS=(
  "baseline|-|"
  "socket_ifname|NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3|"
  "gdr_level|NCCL_NET_GDR_LEVEL=PHB|"
  "guide_pair|NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB|"
  "runtime_connect_off|NCCL_RUNTIME_CONNECT=0|"
  "nchannels_8|NCCL_MAX_NCHANNELS=8|"
  "nchannels_4|NCCL_MAX_NCHANNELS=4|"
  "nchannels_16|NCCL_MAX_NCHANNELS=16|"
  "nchannels_per_peer|NCCL_NCHANNELS_PER_NET_PEER=1|"
  "cxi_cq_and_sw_match|FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RX_MATCH_MODE=software|"
  "proto_simple|NCCL_PROTO=Simple|"
  "cpu_bind|-|--cpu-bind=v,mask_cpu=${CPU_BIND_MASKS}"
  "net_socket|NCCL_NET=Socket|"
)

# Rung 1 is the cheapest decisive cut: does it reproduce, does pinning the interface
# alone fix it, and does the reported workaround fix it.
RUNG1="baseline socket_ifname nchannels_8"
# Interface evidence must come from a separate run: NCCL_DEBUG=INFO writes per-rank logs
# from every rank and would contaminate the very timings we are measuring.
DEBUG_SET="baseline socket_ifname"

case "${MODE}" in
  probe) SELECTED="${RUNG1}" ;;
  sweep) SELECTED="" ;;
  debug) SELECTED="${DEBUG_SET}" ;;
  *) echo "ERROR: unknown MODE=${MODE} (expected probe, sweep or debug)" >&2; exit 2 ;;
esac
[ -n "${VARIANTS_ONLY}" ] && SELECTED="${VARIANTS_ONLY}"

echo "Mode: ${MODE}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}  tasks/node: ${SLURM_NTASKS_PER_NODE:-8}  world: ${SLURM_NPROCS}"
echo "Container: ${CONTAINER}"
echo "Stall timeout: ${STALL_TIMEOUT_S}s per phase"

# The LAIFS subtree carries both the bindings module and the container. When that mount
# stalls, module load blocks forever and the job burns its whole allocation with an
# empty log -- and, worse for this case, a LAIFS stall is indistinguishable from the
# startup stall we are trying to measure. LAIFS can legitimately take minutes under
# load, so allow that but give up rather than hang.
LAIFS_PROBE_TIMEOUT_S="${LAIFS_PROBE_TIMEOUT_S:-300}"
for path in /appl/local/laifs/modules "$(dirname "${CONTAINER}")"; do
  echo "probing ${path} (up to ${LAIFS_PROBE_TIMEOUT_S}s)..."
  if ! timeout "${LAIFS_PROBE_TIMEOUT_S}" ls "${path}" >/dev/null 2>&1; then
    echo "ERROR: ${path} is unreachable (LAIFS mount stalled). Aborting." >&2
    exit 1
  fi
done

module load Local-LAIF lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_runtime/${SLURM_JOB_ID}"
RESULTS_HOST="${WORKDIR}/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${RESULTS_HOST}"
echo "Results: ${RESULTS_HOST}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

# MIOpen defaults to a fixed path under $TMPDIR that the first user on a node ends up
# owning; pin it per user, as the guide does. /tmp is node-local so the directories
# must exist on every node of the allocation.
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export WORLD_SIZE="${SLURM_NPROCS}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export RESULTS_DIR="/work/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"
export STALL_TIMEOUT_S TENSOR_MIB

# Capture the environment once, before any variant has moved anything.
srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  bash /work/repro/rccl_startup_gfx90a/collect_env.sh || true

for row in "${VARIANTS[@]}"; do
  name="${row%%|*}"
  rest="${row#*|}"
  var_env="${rest%%|*}"
  srun_extra="${rest#*|}"

  if [ -n "${SELECTED}" ] && ! grep -qw "${name}" <<< "${SELECTED}"; then
    continue
  fi

  echo
  echo "=== variant ${name} ==="

  # A fresh port per variant, so a lingering rendezvous cannot bleed across runs.
  export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"

  # Clear everything a previous row may have set, so one variable really does move.
  unset NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_RUNTIME_CONNECT NCCL_MAX_NCHANNELS \
        NCCL_NCHANNELS_PER_NET_PEER NCCL_PROTO NCCL_NET \
        FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RX_MATCH_MODE
  if [ "${var_env}" != "-" ] && [ -n "${var_env}" ]; then
    for kv in ${var_env}; do export "${kv?}"; done
  fi

  if [ "${MODE}" = "debug" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
    export NCCL_DEBUG_FILE="/runtime/nccl_${name}_rank%d.log"
  else
    export NCCL_DEBUG=WARN
    unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE
  fi

  echo "env: ${var_env}   srun_extra: ${srun_extra:-none}"

  # kill-on-bad-exit=0 plus || true: a stalled or failed variant is a recorded result,
  # never a fatal job abort. rccl_probe.py exits 75 on a stall, having written its JSON.
  # shellcheck disable=SC2086
  srun --kill-on-bad-exit=0 ${srun_extra} \
    singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    bash /work/repro/rccl_startup_gfx90a/in_container_probe.sh "${name}" || true
done

echo
echo "=== summary ==="
srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  python3 /work/repro/rccl_startup_gfx90a/summarize_sweep.py \
  --results-dir "${RESULTS_DIR}" || true
