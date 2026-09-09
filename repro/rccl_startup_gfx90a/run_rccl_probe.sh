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
#   MODE=combos sbatch ...                                            # what rescues the GDR hang
#   REPEATS=5 VARIANTS_ONLY="baseline" sbatch ...                     # is a stall reproducible?
#   VARIANTS_ONLY="baseline socket_ifname" sbatch ...                 # pick rows
#   MANY_COMMS=8 sbatch ...                                           # communicator-count scaling
#   MODE=upstream sbatch ...                                          # mitigations from recipes#30
#   MODE=hpe sbatch ...                                               # HPE's recommended set
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
MANY_COMMS="${MANY_COMMS:-0}"
# Repeat each selected variant this many times. The fresh_world_first stalls seen in
# jobs 21790392/21790393 hit different variants at different scales, which is the
# signature of a race rather than a setting. One run cannot tell those apart.
REPEATS="${REPEATS:-1}"
# CANARY=1 re-enables the between-variant health check. It currently reports false
# positives (see FINDINGS.md, job 21794113) so it is off; set CANARY_LOG=<path> to
# capture its own error output when diagnosing it.
CANARY="${CANARY:-0}"
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

# Deliberate multi-variable rows, added only after gdr_level was confirmed to hang on
# its own (jobs 21790392, 21790393). Each asks whether one of the reporter's remedies
# rescues that specific hang -- which is the question "why does capping channels help?"
# stated precisely enough to answer.
# Mitigations named in laifs-container-recipes#30 that this investigation had not tried,
# because the tracker was not checked until late. Both were reported there as resolving
# hangs on real LUMI tickets.
UPSTREAM=(
  "cxi_no_host_register|FI_CXI_DISABLE_HOST_REGISTER=1|"
  "mr_cache_monitor|FI_MR_CACHE_MONITOR=userfaultfd|"
  "cxi_both|FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd|"
)

# The full set HPE recommends for RCCL workloads, as relayed by LUMI support, whose
# origin is Samuel Antao's "Extreme Scale AI" talk (Move your AI to LUMI, June 2026).
# Support identifies FI_MR_CACHE_MONITOR=userfaultfd as the crucial one and says the rest
# are recommended but optional -- so both are measured, the single variable and the whole
# set, to see whether the remainder buys anything.
HPE_FULL="HSA_FORCE_FINE_GRAIN_PCIE=1 FI_MR_CACHE_MONITOR=userfaultfd \
FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 \
FI_CXI_RDZV_GET_MIN=0 FI_CXI_DEFAULT_TX_SIZE=2048 NCCL_CROSS_NIC=1 \
FI_CXI_RX_MATCH_MODE=hybrid"
HPE=(
  "hpe_full|${HPE_FULL}|"
  "hpe_minus_monitor|HSA_FORCE_FINE_GRAIN_PCIE=1 FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_DEFAULT_TX_SIZE=2048 NCCL_CROSS_NIC=1 FI_CXI_RX_MATCH_MODE=hybrid|"
)

COMBOS=(
  "gdr_cap|NCCL_NET_GDR_LEVEL=PHB NCCL_MAX_NCHANNELS=8|"
  "gdr_runtime_connect|NCCL_NET_GDR_LEVEL=PHB NCCL_RUNTIME_CONNECT=0|"
  "gdr_ifname|NCCL_NET_GDR_LEVEL=PHB NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3|"
  "gdr_socket_net|NCCL_NET_GDR_LEVEL=PHB NCCL_NET=Socket|"
)
VARIANTS+=("${COMBOS[@]}")
VARIANTS+=("${UPSTREAM[@]}")
VARIANTS+=("${HPE[@]}")

# Rung 1 is the cheapest decisive cut: does it reproduce, does pinning the interface
# alone fix it, and does the reported workaround fix it.
RUNG1="baseline socket_ifname nchannels_8"
# Interface evidence must come from a separate run: NCCL_DEBUG=INFO writes per-rank logs
# from every rank and would contaminate the very timings we are measuring.
DEBUG_SET="baseline socket_ifname gdr_level"
# The GDR hang is the one worth tracing, so combos get their own mode.
COMBO_SET="baseline gdr_level gdr_cap gdr_runtime_connect gdr_ifname gdr_socket_net"
# The upstream-suggested mitigations, against a plain baseline.
UPSTREAM_SET="baseline cxi_no_host_register mr_cache_monitor cxi_both"
# hpe_minus_monitor isolates whether the crucial variable really is the monitor.
HPE_SET="baseline mr_cache_monitor hpe_full hpe_minus_monitor"

case "${MODE}" in
  probe) SELECTED="${RUNG1}" ;;
  sweep) SELECTED="" ;;
  debug) SELECTED="${DEBUG_SET}" ;;
  combos) SELECTED="${COMBO_SET}" ;;
  upstream) SELECTED="${UPSTREAM_SET}" ;;
  hpe) SELECTED="${HPE_SET}" ;;
  *) echo "ERROR: unknown MODE=${MODE} (expected probe, sweep, debug, combos, upstream or hpe)" >&2; exit 2 ;;
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
export STALL_TIMEOUT_S TENSOR_MIB MANY_COMMS

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
  # Clear everything a previous row may have set, so one variable really does move.
  unset NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_RUNTIME_CONNECT NCCL_MAX_NCHANNELS \
        NCCL_MIN_NCHANNELS NCCL_NCHANNELS_PER_NET_PEER NCCL_PROTO NCCL_NET \
        FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RX_MATCH_MODE \
        FI_CXI_DISABLE_HOST_REGISTER FI_MR_CACHE_MONITOR \
        HSA_FORCE_FINE_GRAIN_PCIE FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE \
        FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN FI_CXI_DEFAULT_TX_SIZE NCCL_CROSS_NIC
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

  for rep in $(seq 1 "${REPEATS}"); do
    # Each repeat is recorded under its own name so nothing is overwritten and the
    # summary shows the spread across attempts rather than only the last one.
    if [ "${REPEATS}" -gt 1 ]; then
      run_name="${name}-r${rep}"
      echo "--- repeat ${rep}/${REPEATS} ---"
    else
      run_name="${name}"
    fi
    # A fresh rendezvous port per attempt, so one hung attempt cannot poison the next.
    export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"

    # kill-on-bad-exit=0 plus || true: a stalled or failed variant is a recorded result,
    # never a fatal job abort. rccl_probe.py exits 75 on a stall, having written its JSON.
    # shellcheck disable=SC2086
    srun --kill-on-bad-exit=0 ${srun_extra} \
      singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
      bash /work/repro/rccl_startup_gfx90a/in_container_probe.sh "${run_name}" || true

    # MANDATORY, and the reason job 21790393 produced 11 invalid rows: because
    # --kill-on-bad-exit=0 keeps surviving ranks alive when one aborts on a stall,
    # ranks from a stalled attempt stay blocked inside RCCL, holding their GCDs, while
    # the next attempt starts. The next attempt then stalls too -- in world_second, a
    # warm collective that cannot fail on its own -- and the cascade looks like a
    # finding. Reap everything before measuring anything else.
    srun --overlap --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
      bash -c 'pkill -f rccl_probe.py 2>/dev/null; pkill -f in_container_probe 2>/dev/null; exit 0' || true
    sleep 10

    # Canary: a plain baseline run must stay healthy between variants. If it does not,
    # the node set is contaminated and every later row is suspect -- say so in the log
    # rather than emitting numbers that look like measurements.
    if [ "${CANARY}" = "1" ] && [ "${run_name}" != "canary" ]; then
      srun --kill-on-bad-exit=0 --ntasks="${SLURM_NPROCS}" \
        singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
        bash -c 'RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID \
          HIP_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0} HOME=/runtime \
          timeout 90 python3 -c "
import os, torch, torch.distributed as dist
torch.cuda.set_device(0)
dist.init_process_group(backend=\"nccl\")
t = torch.ones(1024, device=\"cuda:0\")
dist.all_reduce(t)
torch.cuda.synchronize()
dist.destroy_process_group()
"' >"${CANARY_LOG:-/dev/null}" 2>&1 \
        && echo "  canary after ${run_name}: OK" \
        || echo "  canary after ${run_name}: FAILED -- node set contaminated, later rows are NOT valid"
      srun --overlap --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
        bash -c 'pkill -f "python3 -c" 2>/dev/null; exit 0' || true
    fi
  done
done

echo
echo "=== summary ==="
srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  python3 /work/repro/rccl_startup_gfx90a/summarize_sweep.py \
  --results-dir "${RESULTS_DIR}" || true
