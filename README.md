# Serving LLMs on LUMI with vLLM

Minimal runbook to launch vLLM on LUMI, query it with `demo_agent.py`, and run simple benchmarks.

Run commands from the repository root on LUMI:

```bash
cd /scratch/project_462000131/$USER/lumi-ai-assistant-demo
module load cray-python
```

## Prerequisites

Edit the Slurm account and container path in the launcher scripts if needed:

```bash
run_vllm_demo.sh
run_vllm_demo_multinode.sh
```

If the model is not already cached, export a Hugging Face token before submitting:

```bash
export HF_TOKEN=<your-token>
```

## Single Node

Start vLLM:

```bash
sbatch run_vllm_demo.sh
```

Set the job id once:

```bash
JOBID=<jobid>
```

Query the server:

```bash
srun --jobid "$JOBID" --overlap --export=ALL \
  python3 demo_agent.py \
  --base-url http://127.0.0.1:8000/v1 \
  --question "How do I request 1 GPU on LUMI?" \
  --max-tokens 512 \
  --timeout 300
```

## Multi Node

Start vLLM:

```bash
sbatch run_vllm_demo_multinode.sh
```

Set the job id once:

```bash
JOBID=<jobid>
```

Set the head node once:

```bash
NODELIST=$(squeue -j "$JOBID" -h -o %N)
HEAD_NODE=$(scontrol show hostnames "$NODELIST" | head -n1)
echo "$HEAD_NODE"
```

Query the server from the head node:

```bash
srun --jobid "$JOBID" --overlap --exact -N1 -n1 -w "$HEAD_NODE" --export=ALL \
  python3 demo_agent.py \
  --base-url http://127.0.0.1:8000/v1 \
  --question "How do I request 1 GPU on LUMI?" \
  --max-tokens 512 \
  --timeout 300
```

## Benchmarks

Single-node benchmark:

```bash
BENCH_PROFILE=small benchmarks/run_benchmark.sh "$JOBID" 40 4 128
for c in 1 2 4 8 16 32; do
  BENCH_PROFILE=small benchmarks/run_benchmark.sh "$JOBID" 120 "$c" 128
done

python3 benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile small
```

Multi-node benchmark:

```bash
SRUN_NODELIST="$HEAD_NODE" BENCH_PROFILE=large benchmarks/run_benchmark.sh "$JOBID" 40 4 128
SRUN_NODELIST="$HEAD_NODE" BENCH_PROFILE=large benchmarks/run_saturation.sh "$JOBID" 120 128 "8 16 32 64 128"

python3 benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile large
```

## Common Overrides

Set these at submit time when needed:

```bash
MODEL=<model-id-or-local-path> \
TP_SIZE=<tensor-parallel-size> \
PP_SIZE=<pipeline-parallel-size> \
DP_LOCAL_SIZE=<data-parallel-ranks-per-node> \
EXTRA_VLLM_ARGS="--max-model-len 32768" \
sbatch run_vllm_demo_multinode.sh
```

The multi-node default uses:

```bash
MODEL=deepseek-ai/DeepSeek-R1-0528
TP_SIZE=2
PP_SIZE=1
DP_LOCAL_SIZE=$((SLURM_GPUS_ON_NODE / TP_SIZE / PP_SIZE))
DP_SIZE=$((DP_LOCAL_SIZE * SLURM_NNODES))
EXTRA_VLLM_ARGS="--enable-expert-parallel --all2all-backend deepep_low_latency"
```

## Logs

Slurm logs:

```bash
demo-<jobid>.out
demo-<jobid>.err
demo-mn-<jobid>.out
demo-mn-<jobid>.err
```

vLLM logs:

```bash
/scratch/project_462000131/$USER/vllm_runtime/<jobid>/vllm_server.log
/scratch/project_462000131/$USER/vllm_runtime/<jobid>/vllm_server_rank*.log
```
