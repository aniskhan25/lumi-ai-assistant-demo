#!/bin/bash
#SBATCH --job-name=dsv4-fp8-bench
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:30:00
#SBATCH --output=dsv4-fp8-bench-%j.out

# V4-Flash dense FP8 linears on one GCD: stock vs tuned vs bf16 (bench_fp8_linear.py)
#   sbatch repro/dsv4_flash_debug/run_fp8_linear_bench.sh     # from the repo root
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
export AMDGCN_USE_BUFFER_OPS=0   # as served
singularity exec "${CONTAINER}" python "${REPO}/repro/dsv4_flash_debug/bench_fp8_linear.py" \
  "${REPO}/repro/dsv4_flash_debug/fp8_configs" 2>&1 | grep -vE 'amdsmi|raw_cnt|W8A8 Block FP8 kernel config'
