#!/bin/bash
#SBATCH --job-name=dsv41-dev
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=03:00:00
#SBATCH --output=dsv41-dev-%j.out
# Holds one GCD for bring-up; run steps with dev.sh (srun --jobid --overlap).
sleep infinity
