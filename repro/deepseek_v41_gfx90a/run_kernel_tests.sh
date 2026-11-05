#!/bin/bash
#SBATCH --job-name=dsv4-kernels
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:45:00
#SBATCH --output=dsv4-kernels-%j.out

# Phase 0 for DeepSeek-V4.1-Flash: do the DeepSeek-V4 family's ROCm kernels (sparse MLA
# prefill/decode, compressor KV cache, fused indexer q/rope quant, fused q-norm/rope/KV
# insert, inverse rope + FP8 quant) work on one MI250X GCD (gfx90a)? These are vLLM 0.26's
# own tests (vendor/tests_v026, unmodified) run against the container's matching vLLM 0.26,
# so no shims are involved.
#
#   sbatch repro/deepseek_v41_gfx90a/run_kernel_tests.sh          # from the repo root
#   PYTEST_ARGS="-k decode" sbatch repro/deepseek_v41_gfx90a/run_kernel_tests.sh
set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}/repro/deepseek_v41_gfx90a"
RESULTS="${SLURM_SUBMIT_DIR:-$(pwd)}/benchmarks/results/dsv4-kernels/job_${SLURM_JOB_ID}"
mkdir -p "${RESULTS}"

module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
# A per-job Triton cache, so a compile failure is not hidden by an earlier success.
export TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}"

# One process per test file: a hard GPU fault (which kills the process without a pytest
# summary) then names its file and does not take the other files down with it.
export PYTHONFAULTHANDLER=1
for t in "${HERE}"/vendor/tests_v026/test_*.py; do
  name="$(basename "${t}" .py)"
  echo "=== ${name}"
  singularity exec "${CONTAINER}" bash -c "
    cd '${HERE}/vendor' && timeout 1200 python -m pytest -p no:cacheprovider -v -rfEs \
      --junitxml='${RESULTS}/${name}.xml' ${PYTEST_ARGS:-} tests_v026/${name}.py
  " > "${RESULTS}/${name}.log" 2>&1
  echo "exit=$? $(grep -aE '^(=+ .*(passed|failed|error|skipped).* =+)$' "${RESULTS}/${name}.log" | tail -1)"
  grep -aE 'FAILED|ERROR|Fatal Python error|fault|Segmentation|Aborted' "${RESULTS}/${name}.log" | head -5
done
