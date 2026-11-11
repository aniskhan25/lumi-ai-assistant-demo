#!/bin/bash
#SBATCH --job-name=dsv41-tiny
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=01:00:00
#SBATCH --output=dsv41-tiny-%j.out
# Tiny random-weight V4.1 on a full node through the plugin:
#   TP=8 sbatch repro/deepseek_v41_gfx90a/run_tiny.sh        (EAGER= for CUDA graphs)
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR}/repro/deepseek_v41_gfx90a"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
SHM="/tmp/shm-${SLURM_JOB_ID}"; mkdir -p "${SHM}"
singularity exec -B "${SHM}:/dev/shm" --env "PYTHONPATH=${HERE}:/opt/venv/lib/python3.12/site-packages" \
  "${CONTAINER}" bash "${HERE}/tiny_serve.sh"
