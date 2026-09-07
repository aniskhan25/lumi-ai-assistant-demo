#!/bin/bash
#SBATCH --job-name=rccl-bandwidth
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --output=rccl-bw-%j.out
#SBATCH --error=rccl-bw-%j.err

# Measure what capping RCCL channels costs in collective bandwidth -- the reporter's
# third question, and the one they need in order to decide for themselves.
#
#   sbatch repro/rccl_startup_gfx90a/run_bandwidth.sh
#   sbatch --nodes=4 repro/rccl_startup_gfx90a/run_bandwidth.sh
#   sbatch --nodes=8 repro/rccl_startup_gfx90a/run_bandwidth.sh
#   BASE_ENV="NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3" sbatch ...   # measure on the fixed baseline
#
# Runs in the guide's 8-tasks-per-node layout with its CPU bind mask, because a
# bandwidth number taken without CPU-GPU affinity would understate the machine.
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
MAX_BYTES="${MAX_BYTES:-2147483648}"
REPS="${REPS:-20}"
WARMUP="${WARMUP:-5}"
OPS="${OPS:-all_reduce,all_gather,reduce_scatter}"
# Applied to every variant, so the cost of capping is measured on top of whatever
# rungs 1-2 established as the correct baseline rather than on top of a broken one.
BASE_ENV="${BASE_ENV:-}"

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# name | extra env. The uncapped run must be named default_channels: the summariser
# uses it as the reference for the cost column.
VARIANTS=(
  "default_channels|-"
  "nchannels_16|NCCL_MAX_NCHANNELS=16"
  "nchannels_8|NCCL_MAX_NCHANNELS=8"
  "nchannels_4|NCCL_MAX_NCHANNELS=4"
)

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
echo "Nodes: ${SLURM_JOB_NUM_NODES}  world: ${SLURM_NPROCS}  base env: ${BASE_ENV:-none}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
  mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export WORLD_SIZE="${SLURM_NPROCS}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"
export RESULTS_DIR="/work/repro/rccl_startup_gfx90a/results/job_${SLURM_JOB_ID}"
export NCCL_DEBUG=WARN

for row in "${VARIANTS[@]}"; do
  name="${row%%|*}"
  var_env="${row#*|}"

  echo
  echo "=== variant ${name} ==="
  export MASTER_PORT="$(( 20000 + (SLURM_JOB_ID % 10000) + RANDOM % 1000 ))"

  unset NCCL_MAX_NCHANNELS
  for kv in ${BASE_ENV}; do export "${kv?}"; done
  if [ "${var_env}" != "-" ] && [ -n "${var_env}" ]; then
    for kv in ${var_env}; do export "${kv?}"; done
  fi

  srun --kill-on-bad-exit=0 --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" \
    singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    bash -c "export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
      HIP_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-0} HOME=/runtime; \
      python3 /work/repro/rccl_startup_gfx90a/bandwidth_sweep.py \
        --variant '${name}' --results-dir '${RESULTS_DIR}' \
        --max-bytes ${MAX_BYTES} --reps ${REPS} --warmup ${WARMUP} --ops '${OPS}'" || true
done

echo
echo "=== summary ==="
srun --ntasks=1 --ntasks-per-node=1 singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  python3 /work/repro/rccl_startup_gfx90a/summarize_bandwidth.py \
  --results-dir "${RESULTS_DIR}" || true
