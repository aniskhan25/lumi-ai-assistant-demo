#!/bin/bash
#SBATCH --job-name=dsv4-fp8-round
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00
#SBATCH --output=dsv4-fp8-round-%j.out

# Fused FP8 activation rounding vs vLLM quantiser + dequant: exactness and timing (one GCD)
#   sbatch repro/dsv4_flash_debug/run_fp8_round_test.sh          # from the repo root
set -euo pipefail
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SNAPSHOT="$(ls -d /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/* | head -n1)"
module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
# As served (hc_head_triton is a Triton kernel); TileLang does not use this knob.
export AMDGCN_USE_BUFFER_OPS=0
singularity exec "${CONTAINER}" python "${REPO}/repro/dsv4_flash_debug/test_fp8_round.py" 2>&1 \
  | grep -vE 'amdsmi|raw_cnt'
