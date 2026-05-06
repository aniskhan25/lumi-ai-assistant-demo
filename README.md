# LUMI vLLM Demo Runbook

Minimal runbook to launch vLLM on LUMI (single-node or multi-node), query it, and benchmark it.

## 1) Prerequisites
- LUMI GPU project/account access
- Working container path in launcher scripts
- Model path available on compute nodes
- Recommended once per shell:
  - `module use /appl/local/csc/modulefiles/`
  - `module load pytorch`

Notes:
- Scripts also try to auto-load `pytorch` if host Python is missing.
- Run commands from repo root: `/scratch/project_462000131/<user>/lumi-ai-assistant-demo`
- Benchmark outputs under `benchmarks/results/` are generated artifacts (ignored by git).

## 2) Single-Node vLLM
1. Edit `run_vllm_demo.sh`:
   - `CONTAINER`
   - `MODEL`
   - `#SBATCH --account` (and other Slurm settings if needed)
2. Submit:
   - `sbatch run_vllm_demo.sh`
3. Save job id:
   - `JOBID=<jobid>`
4. Query:
   - `srun --jobid "$JOBID" --overlap --export=ALL python demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on LUMI?"`

Single-node multi-GPU example (TP=4):
- `sbatch --nodes=1 --gpus-per-node=4 --export=ALL,TP_SIZE=4,STARTUP_TIMEOUT_S=1800 run_vllm_demo.sh`

## 3) Multi-Node vLLM
1. Edit `run_vllm_demo_multinode.sh`:
   - `CONTAINER`
   - `MODEL`
   - `#SBATCH --account`
2. Submit (example 2 nodes x 4 GPUs):
   - `sbatch --nodes=2 --gpus-per-node=4 --export=ALL,TP_SIZE=4,PP_SIZE=2,MASTER_PORT=29501,STARTUP_TIMEOUT_S=2400 run_vllm_demo_multinode.sh`
3. Save job id:
   - `JOBID=<jobid>`
4. Get NodeList and head node:
   - `NODELIST=$(squeue -j "$JOBID" -h -o %N)`
   - `HEAD_NODE=$(scontrol show hostnames "$NODELIST" | head -n1)`
   - `echo "NODELIST=$NODELIST"`
   - `echo "HEAD_NODE=$HEAD_NODE"`
5. Query (must be pinned to head node):
   - `srun --jobid "$JOBID" --overlap -w "$HEAD_NODE" --export=ALL python demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on LUMI?"`

## 4) Benchmarks
### Single-node
- `JOBID=<jobid>`
- `PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 40 4 128`
- `for c in 1 2 4 8 16 32; do PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 120 "$c" 128; done`

### Multi-node (pin benchmark step to head node)
- `JOBID=<jobid>`
- `HEAD_NODE=<head-node>`
- `SRUN_NODELIST="$HEAD_NODE" PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 40 4 128`
- `SRUN_NODELIST="$HEAD_NODE" PYTHON_BIN=python benchmarks/run_saturation.sh "$JOBID" 120 128 "8 16 32 64 128"`

If startup is slow, increase benchmark readiness wait:
- `STARTUP_WAIT_S=600 STARTUP_POLL_S=2 PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 40 4 128`

### Summarize
- `python benchmarks/summarize_results.py --job-dir benchmarks/results/job_$JOBID`

## 5) Logs and Debugging
Slurm logs:
- `demo-<jobid>.out`
- `demo-<jobid>.err`
- `demo-mn-<jobid>.out`
- `demo-mn-<jobid>.err`

Runtime logs:
- Single-node: `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server.log`
- Multi-node:
  - `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/srun_step.log`
  - `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/launcher_rank*.log`
  - `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server_rank*.log`

## 6) Supplemental: Puhti
Use `run_vllm_demo_puhti.sh`.

- Load module: `module load pytorch/2.9`
- Submit: `sbatch run_vllm_demo_puhti.sh`
- Optional overrides at submit time:
  - `sbatch --export=ALL,MODEL=/scratch/project_2014553/<user>/models/<model-dir>,TP_SIZE=1 run_vllm_demo_puhti.sh`
- Query:
  - `srun --jobid <jobid> --overlap --export=ALL python3 demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on Puhti?"`
