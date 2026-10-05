#!/bin/bash
#SBATCH --job-name=hf-download
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=hf-download-%j.out

# Download a Hugging Face model into the shared per-user cache on a CPU node, so a GPU
# allocation never spends its startup timeout on a download.
#   sbatch repro/download_model.sh Qwen/Qwen3-Coder-480B-A35B-Instruct
set -euo pipefail

MODEL="${1:?usage: sbatch repro/download_model.sh <model-id>}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache"

# Without the bindings /scratch (really /pfs/lustrep4) is read-only inside the container.
module load Local-LAIF lumi-aif-singularity-bindings
singularity exec "${CONTAINER}" hf download "${MODEL}" --max-workers 16
du -sh "${HF_HOME}/hub/models--${MODEL//\//--}"
