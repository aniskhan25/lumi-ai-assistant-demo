#!/bin/bash
#SBATCH --job-name=skinny-strides
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:15:00
#SBATCH --output=skinny-strides-%j.out
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
singularity exec "${CONTAINER}" python "${REPO}/repro/dsv4_flash_debug/test_skinny_gemm_strides.py" 2>&1 | grep -vE 'amdsmi|raw_cnt'
