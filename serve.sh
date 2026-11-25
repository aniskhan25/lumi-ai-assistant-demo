#!/bin/bash
#SBATCH --job-name=dsv41-serve
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00
#SBATCH --output=dsv41-serve-%j.out
# DeepSeek-V4.1-Flash on one LUMI-G node (8 GCDs) with the lumi_dsv41 plugin and the best
# measured settings, from the image build.sh makes. Usage (from the repository root):
#
#   MODEL=/path/to/DeepSeek-V4.1-Flash sbatch serve.sh [extra vllm serve arguments]
#
# The OpenAI-compatible API listens on port ${PORT:-8000} of the compute node.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL="${MODEL:-/flash/${SLURM_JOB_ACCOUNT}/${USER}/DeepSeek-V4.1-Flash}"
IMAGE="${IMAGE:-${REPO}/lumi-dsv41.sif}"
PORT="${PORT:-8000}"
[ -f "${MODEL}/config.json" ] || { echo "MODEL=${MODEL}: no config.json"; exit 1; }
[ -f "${IMAGE}" ] || { echo "IMAGE=${IMAGE}: not found; build it with: sbatch build.sh"; exit 1; }

module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
export VLLM_USE_V2_MODEL_RUNNER=1        # required: Engram state lives in Model Runner V2
export VLLM_USE_BREAKABLE_CUDAGRAPH=1    # CUDA graphs around the plugin's attention breaks
export AMDGCN_USE_BUFFER_OPS=0           # skips a Triton 3.6 AMD pass that fails on the fused q/kv RMSNorm kernel
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}" MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
# A private /dev/shm: on shared nodes another job's epilog can wipe the node-wide one,
# which kills vLLM's shared-memory broadcast.
SHM="/tmp/shm-${SLURM_JOB_ID}"; mkdir -p "${SHM}"

exec singularity exec -B "${SHM}:/dev/shm" "${IMAGE}" \
  vllm serve "${MODEL}" --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 \
    --moe-backend triton_unfused \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --enable-prompt-tokens-details \
    "$@"
