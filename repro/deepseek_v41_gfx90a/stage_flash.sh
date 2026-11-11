#!/bin/bash
#SBATCH --job-name=dsv41-stage
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=dsv41-stage-%j.out
# Copy the V4.1 snapshot from scratch to LUMI-F (flash). From scratch, 8 TP ranks each
# reading the 476 GB checkpoint took >1 h: the page cache cannot hold it next to the
# 189 GiB of pinned Engram tables, so every rank goes back to Lustre.
set -euo pipefail
SRC="$(ls -d /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/* | head -n1)"
DST="/flash/${SLURM_JOB_ACCOUNT}/${USER}/DeepSeek-V4.1-Flash"
mkdir -p "${DST}"
cd "${SRC}"
find . -type d -exec mkdir -p "${DST}/{}" \;
# -L: the snapshot holds symlinks into blobs/. Big files first so the parallel tail is short.
find . \( -type f -o -type l \) -printf '%s %p\n' | sort -rn | cut -d' ' -f2- \
  | xargs -P 16 -I{} cp -L --preserve=timestamps {} "${DST}/{}"
du -sh "${SRC}/" --apparent-size -L; du -sh "${DST}" --apparent-size
echo STAGE_DONE "${DST}"
