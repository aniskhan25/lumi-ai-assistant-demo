# LUMI AI Assistant Demo (vLLM + Local RAG)

This repo is a minimal, local-only HPC demo:
- Start a vLLM OpenAI-compatible server in a Slurm job.
- Query it from another shell with `demo_agent.py`.
- Optionally run repeatable load benchmarks.

## Contents
- `run_vllm_demo.sh`: primary LUMI launcher (persistent vLLM server job)
- `run_vllm_demo_multinode.sh`: LUMI multi-node launcher (vLLM mp backend)
- `demo_agent.py`: CLI agent with simple RAG + Slurm template tool
- `benchmarks/benchmark_openai.py`: OpenAI-compatible benchmark runner
- `benchmarks/run_benchmark.sh`: run one benchmark profile against a running job
- `benchmarks/run_saturation.sh`: run a concurrency sweep
- `benchmarks/summarize_results.py`: summarize benchmark summaries
- `benchmarks/prompts.txt`: prompt set for repeatable runs
- `lumi_docs/`: local docs used for retrieval

## Prerequisites (LUMI)
- GPU allocation on LUMI (`standard-g` or your project partition)
- vLLM-capable Apptainer/Singularity image
- Model path available from compute nodes
- Module Python for query/benchmark commands:
  - `module use /appl/local/csc/modulefiles/`
  - `module load pytorch`

## Quick Start (LUMI)
1. Edit `run_vllm_demo.sh`:
   - `CONTAINER`
   - `MODEL`
   - Slurm directives (`#SBATCH --account`, partition, time, GPU count)
2. Submit:
   - `sbatch run_vllm_demo.sh`
3. Wait until `demo-<jobid>.out` shows:
   - `vLLM server is ready at http://127.0.0.1:8000/v1`
4. Query from another shell:
   - `srun --jobid <jobid> --overlap --export=ALL python /scratch/project_462000131/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on LUMI?"`

## Multi-GPU on LUMI (Single Node)
Key rule: `TP_SIZE <= allocated GPUs`.

Example (4 GPUs, TP=4):
- `sbatch --nodes=1 --gpus-per-node=4 --export=ALL,TP_SIZE=4,STARTUP_TIMEOUT_S=1800 run_vllm_demo.sh`

Useful launcher knobs in `run_vllm_demo.sh`:
- `TP_SIZE` (default `1`)
- `STARTUP_TIMEOUT_S` (default `900`)
- `ROCM_COMPAT_MODE` (default `1`, safer TP>1 on some ROCm stacks)
- `ENFORCE_EAGER` (default `1`)

## Multi-Node Testing on LUMI
Use `run_vllm_demo_multinode.sh` for one vLLM server spanning multiple nodes.

Default script behavior:
- `#SBATCH --nodes=2`
- `#SBATCH --ntasks-per-node=1` (one launcher rank per node)
- `#SBATCH --gpus-per-node=4`
- `TP_SIZE` defaults to GPUs per node
- `PP_SIZE` defaults to number of nodes
- `MASTER_PORT` defaults from job id (can be overridden)

Example submit:
- `sbatch --nodes=2 --gpus-per-node=4 --export=ALL,TP_SIZE=4,PP_SIZE=2,MASTER_PORT=29501,STARTUP_TIMEOUT_S=2400 run_vllm_demo_multinode.sh`

After readiness, query pinned to the reported head node:
- `srun --jobid <jobid> --overlap -w <head_node> --export=ALL python /scratch/project_462000131/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on LUMI?"`

Benchmark on multi-node job (pin to head node so `127.0.0.1` resolves to API node):
- `SRUN_NODELIST=<head_node> PYTHON_BIN=python benchmarks/run_benchmark.sh <jobid> 40 4 128`

## Benchmark Workflow (LUMI)
Run from repo root after the server is ready.

1. Single run:
   - `benchmarks/run_benchmark.sh <jobid> 40 4 128`
2. Concurrency sweep:
   - `for c in 1 2 4 8; do benchmarks/run_benchmark.sh <jobid> 80 "$c" 128; done`
3. Saturation sweep:
   - `benchmarks/run_saturation.sh <jobid> 120 128 "8 10 12 16 20 24"`
4. Summarize:
   - `python benchmarks/summarize_results.py --job-dir benchmarks/results/job_<jobid>`

Notes:
- `benchmarks/run_benchmark.sh` uses `PYTHON_BIN` (default `python3`).
- `benchmarks/run_benchmark.sh` supports `SRUN_NODELIST=<node>` to pin benchmarking to a specific node.
- On LUMI module Python, use:
  - `PYTHON_BIN=python benchmarks/run_benchmark.sh <jobid> 40 4 128`

## Logs
- Slurm out/err: `demo-%j.out`, `demo-%j.err`
- Server log inside container: `/runtime/vllm_server.log`
- Server log on host (LUMI): `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server.log`
- Multi-node per-rank logs (LUMI): `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server_rank*.log`

## Optional Agent Usage
- Single question:
  - `python demo_agent.py --question "How do I request a GPU?"`
- Question file:
  - `python demo_agent.py --question-file examples/sample_questions.md`

## Supplemental: Puhti
Use `run_vllm_demo_puhti.sh` if you want the same demo flow on Puhti.

Puhti-specific prerequisites:
- `module load pytorch/2.9`

Puhti launch/query pattern:
1. `sbatch run_vllm_demo_puhti.sh`
2. Query with overlap step (same pattern as LUMI):
   - `srun --jobid <jobid> --overlap --export=ALL python3 /scratch/project_2014553/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on Puhti?"`

Puhti log path (current script configuration):
- `/scratch/project_2014553/anisrahm/vllm_runtime/<jobid>/vllm_server.log`

## Notes
- Both launchers keep vLLM alive and expect query/benchmark commands from a separate `srun --jobid ... --overlap` step.
- If `MODEL` is a local directory, it is bind-mounted into the container.
- This is a demo only; retrieval docs are minimal and not authoritative.
