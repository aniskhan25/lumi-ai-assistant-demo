#!/bin/bash
#SBATCH --job-name=lumi-vllm-mn-demo
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=460G
#SBATCH --time=02:00:00
#SBATCH --output=demo-mn-%j.out
#SBATCH --error=demo-mn-%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/hf-cache"
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/vllm-cache"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-9999}"
export EXTRA_VLLM_ARGS
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}"

SOCKET_FILE="${TMPDIR}/vllm-${SLURM_JOB_ACCOUNT}.sock"

echo "Launching vLLM:"
echo "  model=${MODEL}"
echo "  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  socket=${SOCKET_FILE}"
echo "  tensor_parallel=${SLURM_GPUS_ON_NODE}, pipeline_parallel=${SLURM_NNODES}"

srun singularity exec "${CONTAINER}" ./launch_vllm_rank.sh "${MODEL}" \
  --tensor-parallel "${SLURM_GPUS_ON_NODE}" \
  --pipeline-parallel "${SLURM_NNODES}" \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency \
  --uds "${SOCKET_FILE}" \
  ${EXTRA_VLLM_ARGS}
