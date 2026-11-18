#!/bin/bash
#SBATCH --job-name=dsv4-moe-matmul
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00
#SBATCH --output=dsv4-moe-matmul-%j.out

# V4 MXFP4 expert GEMMs at decode sizes: default vs forced block_n / split_k (one GCD)
#   sbatch repro/dsv4_flash_debug/run_moe_matmul_test.sh          # from the repo root
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SNAPSHOT="$(ls -d /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/* | head -n1)"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
singularity exec "${CONTAINER}" python "${REPO}/repro/dsv4_flash_debug/test_moe_matmul.py" 2>&1 \
  | grep -vE 'amdsmi|raw_cnt'
