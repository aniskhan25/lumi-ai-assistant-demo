# LUMI/Puhti AI Assistant Demo (vLLM + Local RAG)

This repo is a minimal, local-only demo of an "agent-like" assistant on HPC. It runs a vLLM server and a CLI agent in a single Slurm job, using local docs for retrieval.

## Contents
- `run_vllm_demo.sh`: single-job orchestration for LUMI
- `run_vllm_demo_puhti.sh`: single-job orchestration for Puhti
- `demo_agent.py`: CLI agent with simple RAG + a Slurm template tool
- `benchmarks/benchmark_openai.py`: OpenAI-compatible benchmark runner
- `benchmarks/run_benchmark_puhti.sh`: helper to benchmark against a running Puhti job
- `benchmarks/run_saturation_puhti.sh`: helper for high-concurrency saturation sweep on Puhti
- `benchmarks/prompts_puhti.txt`: prompt set for repeatable benchmark runs
- `benchmarks/summarize_results.py`: summarize and rank benchmark summaries
- `lumi_docs/`: local demo docs used for retrieval
- `examples/sample_questions.md`: demo prompts

## Prerequisites
- Access to a GPU partition on LUMI or Puhti
- An Apptainer image with vLLM installed
- A local or staged model path accessible inside the container

## Quick Start
1. Pick the script for your cluster:
   - LUMI: `run_vllm_demo.sh`
   - Puhti: `run_vllm_demo_puhti.sh`
2. Edit the selected script:
   - Set `CONTAINER` to your vLLM-enabled container path
   - Set `MODEL` to your model path or model identifier
   - Update `#SBATCH` account/partition/GPU/time directives for your project
3. Submit the job:
   - LUMI: `sbatch run_vllm_demo.sh`
   - Puhti: `sbatch run_vllm_demo_puhti.sh`
4. Follow the prompt in `demo-<jobid>.out` to ask questions.

## Query A Running Server
If vLLM is already running inside a Slurm job, run queries from another step with `srun --jobid ... --overlap`.

Puhti example:
`srun --jobid <jobid> --overlap python3 /scratch/project_2014553/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on Puhti?"`

LUMI example:
`srun --jobid <jobid> --overlap python3 /scratch/project_462000131/<user>/lumi-ai-assistant-demo/demo_agent.py --base-url http://127.0.0.1:8000/v1 --question "How do I request 1 GPU on LUMI?"`

Use `python3` in these commands. On some systems `python` is not in `PATH` for `srun` steps.

## Benchmark Plan
Use the same model and same prompt file for all runs. Change one variable at a time.

1. Smoke test:
   - `requests=20`, `concurrency=1`, `max_tokens=64`
2. Concurrency sweep:
   - `requests=80`, `concurrency=1,2,4,8`, `max_tokens=128`
3. Output length sweep:
   - `requests=60`, `concurrency=4`, `max_tokens=64,256,512`
4. Stability check:
   - Repeat one mid-load case (`requests=80`, `concurrency=4`, `max_tokens=128`) 3 times
5. Saturation check:
   - `requests=120`, `concurrency=8,10,12,16`, `max_tokens=128`
6. Plateau check:
   - extend saturation to `concurrency=20,24` (and optionally 28)
   - stop increasing concurrency when throughput gain is small and latency cost is high

Record and compare:
- request throughput (`throughput_req_s`)
- latency (`latency_p50_s`, `latency_p95_s`, `latency_p99_s`)
- token throughput (`throughput_tokens_s`, `throughput_completion_tokens_s`)
- failure rate (`requests_failed`)

## Run Benchmarks
From repo root, against a running Puhti vLLM job:

1. Single run:
   - `benchmarks/run_benchmark_puhti.sh <jobid> 40 4 128`
2. Concurrency sweep:
   - `for c in 1 2 4 8; do benchmarks/run_benchmark_puhti.sh <jobid> 80 "$c" 128; done`
3. Token-length sweep:
   - `for t in 64 256 512; do benchmarks/run_benchmark_puhti.sh <jobid> 60 4 "$t"; done`
4. Saturation sweep:
   - `benchmarks/run_saturation_puhti.sh <jobid> 120 128 "8 10 12 16"`
5. Summarize one job directory:
   - `python3 benchmarks/summarize_results.py --job-dir benchmarks/results/job_<jobid>`
6. Plateau extension:
   - `benchmarks/run_saturation_puhti.sh <jobid> 120 128 "16 20 24"`
7. Optional: tune startup wait if model load is slow:
   - `python3 benchmarks/benchmark_openai.py --base-url http://127.0.0.1:8000/v1 --requests 40 --concurrency 4 --max-tokens 128 --startup-wait-s 300 --startup-poll-s 2`

Results are written to:
- `benchmarks/results/job_<jobid>/summary_*.json`
- `benchmarks/results/job_<jobid>/raw_*.json`

## Plateau Decision Rule
Use this to decide when concurrency is no longer worth increasing.

1. Run at least 3 adjacent points (for example `16,20,24`) with same `requests` and `max_tokens`.
2. Compare each point to the previous point using `throughput_completion_tokens_s` and `latency_p95_s`.
3. Treat it as plateau if both are true:
   - throughput gain is below 5% (`new/old < 1.05`)
   - p95 latency increase is above 10% (`new/old > 1.10`)
4. Also stop if failures appear (`requests_failed > 0`), even if throughput still rises.
5. Pick the highest concurrency before plateau/failures as the production throughput profile.

## Operating Profiles (Puhti, current measurements)
Based on your benchmark summaries with `max_tokens=128`.

1. Throughput profile:
   - `concurrency=16`, `max_tokens=128`
   - observed `throughput_completion_tokens_s=539.216`
   - observed `latency_p95_s=3.590`
   - run: `benchmarks/run_benchmark_puhti.sh <jobid> 120 16 128`
2. Interactive profile:
   - `concurrency=4`, `max_tokens=64`
   - observed `latency_p95_s=1.680`
   - observed `throughput_completion_tokens_s=153.394`
   - run: `benchmarks/run_benchmark_puhti.sh <jobid> 60 4 64`

## What the Demo Does
- Starts a vLLM OpenAI-compatible server bound to `127.0.0.1` only
- Runs `demo_agent.py` which:
  - Retrieves top-k docs from `./lumi_docs`
  - Calls the model via `/v1/chat/completions`
  - Adds a simple Slurm template tool when it detects Slurm-related questions

## Optional: Non-interactive Mode
You can run questions from a file or a single question:
- `python demo_agent.py --question-file examples/sample_questions.md`
- `python demo_agent.py --question "How do I request a GPU?"`

## Tool Template Defaults
The Slurm template tool uses these env vars if set:
- `ACCOUNT`, `PARTITION`, `GPUS`, `HOURS`

If not set, it falls back to placeholders.

## Logs
- Slurm output: `demo-%j.out`
- Slurm error: `demo-%j.err`
- vLLM server log (inside container): `/runtime/vllm_server.log`
- Host path (Puhti): `/scratch/project_2014553/<user>/vllm_runtime/<jobid>/vllm_server.log`
- Host path (LUMI): `/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server.log`

## Notes
- `run_vllm_demo.sh` (LUMI) and `run_vllm_demo_puhti.sh` (Puhti) now use the same minimal flow: start vLLM, wait for `/v1/models`, then run `demo_agent.py`.
- In both scripts, if `MODEL` points to a local directory, it is bind-mounted into the container automatically.
- Both scripts create a per-job runtime/cache directory on scratch and mount it at `/runtime`.
- This is a demo only; the docs in `lumi_docs/` are minimal and not authoritative.
- The retrieval is TF-IDF based and designed to be dependency-light.
