# Slurm Basics (LUMI Demo)

This is a short demo doc for Slurm usage on LUMI. It is not authoritative.

## Common flags
- `--account` or `-A`: Project or account to charge.
- `--partition` or `-p`: Target partition (e.g., `small-g`, `standard-g`).
- `--nodes` and `--ntasks`: Nodes and tasks.
- `--gpus-per-node`: GPU count per node.
- `--time`: Walltime, format `HH:MM:SS`.

## Typical GPU job outline
1. Load modules / activate environment.
2. Set `ROCR_VISIBLE_DEVICES` or rely on Slurm GPU binding.
3. Launch with `srun`.

## Interactive allocation
Use `salloc` for an interactive session, then run your commands on the allocated node.

## Job output
Use `#SBATCH --output=demo-%j.out` and `#SBATCH --error=demo-%j.err` for logs.
