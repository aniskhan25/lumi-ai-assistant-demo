#!/bin/bash
#SBATCH --job-name=dsv41-sparse-split
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00
#SBATCH --output=dsv41-sparse-split-%j.out

# V4.1 split-K sparse decode (DSV41_SPARSE_SPLIT) vs the gfx90a fallback (test_sparse_split.py, one GCD)
#   sbatch repro/deepseek_v41_gfx90a/run_sparse_split_test.sh    # from the repo root
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
export AMDGCN_USE_BUFFER_OPS=0   # as served
singularity exec --env "PYTHONPATH=${REPO}/repro/deepseek_v41_gfx90a:/opt/venv/lib/python3.12/site-packages" "${CONTAINER}" python "${REPO}/repro/deepseek_v41_gfx90a/test_sparse_split.py" 2>&1 \
  | grep -vE 'amdsmi|raw_cnt'
