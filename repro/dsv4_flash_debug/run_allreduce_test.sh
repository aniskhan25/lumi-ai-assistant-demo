#!/bin/bash
#SBATCH --job-name=dsv4-allreduce
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=dsv4-allreduce-%j.out

# vLLM custom all-reduce vs RCCL on the 8 GCDs of one node (test_allreduce.py)
#   sbatch repro/dsv4_flash_debug/run_allreduce_test.sh          # from the repo root
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
singularity exec "${CONTAINER}" python -m torch.distributed.run --nproc-per-node 8 \
  "${REPO}/repro/dsv4_flash_debug/test_allreduce.py" 2>&1 | grep -vE 'amdsmi|raw_cnt'
