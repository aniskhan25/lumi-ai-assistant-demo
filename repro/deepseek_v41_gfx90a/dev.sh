#!/bin/bash
# Run a command in the container on the held dev GCD (hold_gpu.sh), plugin on PYTHONPATH.
#   repro/deepseek_v41_gfx90a/dev.sh <jobid> python repro/deepseek_v41_gfx90a/check_import.py
set -euo pipefail
JOB="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
module load Local-LAIF lumi-aif-singularity-bindings >/dev/null 2>&1
srun --jobid="${JOB}" --overlap --ntasks=1 bash -c 'export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"; exec "$@"' _ \
  singularity exec --env "PYTHONPATH=${HERE}:/opt/venv/lib/python3.12/site-packages" "${CONTAINER}" "$@"
