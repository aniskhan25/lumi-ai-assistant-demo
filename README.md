# LUMI AI Assistant Demo (vLLM + Local RAG)

This repo is a minimal, local-only demo of an "agent-like" assistant on LUMI. It runs a vLLM server and a CLI agent in a single Slurm job, using local docs for retrieval.

## Contents
- `run_vllm_demo.sh`: single-job orchestration
- `demo_agent.py`: CLI agent with simple RAG + a Slurm template tool
- `lumi_docs/`: local demo docs used for retrieval
- `examples/sample_questions.md`: demo prompts

## Prerequisites
- Access to a LUMI GPU partition
- An Apptainer image with vLLM installed
- A local or staged model path accessible inside the container

## Quick Start
1. Edit `run_vllm_demo.sh`:
   - Set `CONTAINER` to your vLLM-enabled container path
   - Set `MODEL` to your model path or identifier
   - Optionally update `--account`, `--partition`, and GPU/time directives
2. Submit the job:
   - `sbatch run_vllm_demo.sh`
3. Follow the prompt in `demo-<jobid>.out` to ask questions.

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
- vLLM server log: `vllm_server.log`

## Notes
- This is a demo only; the docs in `lumi_docs/` are minimal and not authoritative.
- The retrieval is TF-IDF based and designed to be dependency-light.
