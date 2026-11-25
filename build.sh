#!/bin/bash
#SBATCH --job-name=dsv41-build
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=dsv41-build-%j.out
# Build the plugin wheel and the serving image (LAIF lumi-multitorch + the wheel).
# Usage (from the repository root): sbatch build.sh   ->  dist/*.whl, lumi-dsv41.sif
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE="${BASE:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
IMAGE="${IMAGE:-${REPO}/lumi-dsv41.sif}"
cd "${REPO}"

# 1. The wheel, with the base image's Python (pip fetches the build backend, hatchling).
rm -rf dist
singularity exec "${BASE}" python -m pip wheel --no-deps --no-cache-dir -w dist .
WHEEL="$(ls dist/lumi_dsv41-*.whl)"
echo "built ${WHEEL}"

# 2. The image: an unprivileged build in LUMI's proot mode (proot from the systools module).
module load LUMI/25.09 partition/C systools/25.09-2
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/singularity-tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/singularity-cache}"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
singularity build --force --build-arg BASE="${BASE}" --build-arg WHEEL="${WHEEL}" \
  "${IMAGE}" container/lumi-dsv41.def

# 3. Check: the plugin is visible to vLLM in the new image.
singularity exec "${IMAGE}" python -c "
from importlib.metadata import entry_points
import sys
assert 'lumi_dsv41.loader' in sys.modules, 'lumi_dsv41.pth did not run'
eps = [e.value for e in entry_points(group='vllm.general_plugins')]
assert 'lumi_dsv41:register' in eps, eps
print('plugin registered:', eps)"
echo "image ${IMAGE}"
