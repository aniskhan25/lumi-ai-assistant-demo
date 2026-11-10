#!/bin/bash
#SBATCH --job-name=dsv4-ref
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --output=dsv4-ref-%j.out

# Ground truth for the V4-Flash garbage on gfx90a. Runs DeepSeek's own reference
# implementation (reference_upstream/, verbatim) with its TileLang kernels replaced by plain
# PyTorch (reference_torch/), MP=8 on one LUMI-G node, greedy, on the same prompts vLLM
# answered with garbage. If this is coherent, the bug is in vLLM's ROCm path; if not, it is
# lower down (PyTorch/ROCm FP8 casts, RCCL, ...).
#
#   sbatch repro/dsv4_flash_debug/run_reference.sh              # from the repo root
set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
HERE="${REPO}/repro/dsv4_flash_debug"
SNAPSHOT="$(ls -d /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/* | head -n1)"
CKPT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/dsv4-flash-ref-mp8"
LOG="${REPO}/benchmarks/results/dsv4-ref/job_${SLURM_JOB_ID}"
mkdir -p "${LOG}"

module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
SHM="/tmp/shm-${SLURM_JOB_ID}"; mkdir -p "${SHM}"
RUN=(singularity exec -B "${SHM}:/dev/shm" "${CONTAINER}")

# generate.py imports `kernel` from its own directory and the chat encoding from
# ../encoding, so assemble a run directory: upstream model/generate/config, our kernels.
RUNDIR="/tmp/dsv4-ref-${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}/inference"
cp "${HERE}"/reference_upstream/{model.py,generate.py,config.json} "${RUNDIR}/inference/"
cp -r "${HERE}"/reference_torch/{kernel.py,fast_hadamard_transform} "${RUNDIR}/inference/"
ln -sfn "${SNAPSHOT}/encoding" "${RUNDIR}/encoding"

if [ ! -f "${CKPT}/model7-mp8.safetensors" ]; then
  echo "=== converting the HF checkpoint to MP=8 shards ==="
  SECONDS=0
  "${RUN[@]}" python "${HERE}/reference_upstream/convert.py" --hf-ckpt-path "${SNAPSHOT}" \
    --save-path "${CKPT}" --n-experts 256 --model-parallel 8 --expert-dtype fp4
  echo "converted in ${SECONDS}s"
fi

cat > "${LOG}/prompts.txt" <<'EOF'
What is the capital of Finland? Answer with one word.

What is 17 multiplied by 23? Answer with the number only.

Translate to English: 'Sopimus on pätemätön, jos se on tehty pakottamalla.'
EOF

echo "=== reference generation (greedy) ==="
SECONDS=0
"${RUN[@]}" bash -c "cd '${RUNDIR}/inference' && torchrun --nproc-per-node 8 generate.py \
  --ckpt-path '${CKPT}' --config config.json --input-file '${LOG}/prompts.txt' \
  --max-new-tokens 48 --temperature 0" 2>&1 | grep -vE 'amdsmi|raw_cnt' | tee "${LOG}/generate.log"
echo "generated in ${SECONDS}s"
