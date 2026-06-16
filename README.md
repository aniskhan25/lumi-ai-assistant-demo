# Benchmarking LLM Serving on LUMI with vLLM

Minimal runbook to launch vLLM on LUMI and run simple benchmarks.

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

## Benchmarks

Single-node benchmark:

```bash
BENCH_PROFILE=small benchmarks/run_benchmark.sh "$JOBID" 40 4 128
```

```bash
BENCH_PROFILE=small benchmarks/run_saturation.sh "$JOBID" 120 128 "8 16 32 64 96"
```

```bash
python3 benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile small
```

Multi-node benchmark:

```bash
SRUN_NODELIST="$HEAD_NODE" BENCH_PROFILE=large benchmarks/run_benchmark.sh "$JOBID" 40 4 128
```

```bash
SRUN_NODELIST="$HEAD_NODE" BENCH_PROFILE=large benchmarks/run_saturation.sh "$JOBID" 120 128 "8 16 32 64 128"
```

```bash
python3 benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile large
```

For slow models, increase the per-request timeout:

```bash
REQUEST_TIMEOUT_S=600 SRUN_NODELIST="$HEAD_NODE" BENCH_PROFILE=large benchmarks/run_saturation.sh "$JOBID" 120 128 "1 2 4 8 16"
```

## Common Overrides

Set these at submit time when needed:

```bash
MODEL=<model-id-or-local-path> \
TP_SIZE=<gpus-per-node> \
PP_SIZE=<nodes> \
EXTRA_VLLM_ARGS="--max-model-len 32768" \
sbatch run_vllm_demo_multinode.sh
```

The multi-node default uses:

```bash
MODEL=openai/gpt-oss-120b
TP_SIZE=$SLURM_GPUS_ON_NODE
PP_SIZE=$SLURM_JOB_NUM_NODES
EXTRA_VLLM_ARGS="--max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching"
```

## Successful Launch Commands

Single node, 2 GCDs:

```bash
MODEL=Qwen/Qwen2.5-32B-Instruct \
TP_SIZE=2 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--dtype bfloat16 --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90" \
sbatch --gpus-per-node=2 run_vllm_demo.sh
```

Single node, 4 GCDs:

```bash
MODEL=Qwen/Qwen2.5-72B-Instruct \
TP_SIZE=4 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--dtype bfloat16 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90" \
sbatch --gpus-per-node=4 run_vllm_demo.sh
```

Single node, 8 GCDs:

```bash
STARTUP_TIMEOUT_S=1800 \
MODEL=mistralai/Mixtral-8x22B-Instruct-v0.1 \
TP_SIZE=8 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--dtype bfloat16 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90" \
sbatch --gpus-per-node=8 run_vllm_demo.sh
```

Two full nodes:

```bash
STARTUP_TIMEOUT_S=2700 \
MODEL=deepseek-ai/DeepSeek-R1-0528 \
TP_SIZE=8 \
PP_SIZE=2 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--enable-expert-parallel --all2all-backend deepep_high_throughput --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95" \
sbatch run_vllm_demo_multinode.sh
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

## Benchmark Results

| Scenario | Model | Resources | Best concurrency | p95 latency (s) | Completion throughput (tokens/s) | Completion throughput/GCD (tokens/s) |
|---|---|---:|---:|---:|---:|---:|
| Single GCD default | `mistralai/Mistral-7B-Instruct-v0.2` | 1 GCD | 32 | 2.923 | 1340.821 | 1340.821 |
| Multi-node default | `openai/gpt-oss-120b` | 2 nodes, 16 GCDs | 128 | 10.395 | 1473.234 | 92.077 |
| Single node, 2 GCDs | `Qwen/Qwen2.5-32B-Instruct` | 2 GCDs | 64 | 26.397 | 293.506 | 146.753 |
| Single node, 4 GCDs | `Qwen/Qwen2.5-72B-Instruct` | 4 GCDs | 64 | 3.336 | 2319.813 | 579.953 |
| Single node, 8 GCDs | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 8 GCDs | 96 | 25.226 | 455.677 | 56.960 |
| Two full nodes | `deepseek-ai/DeepSeek-R1-0528` | 2 nodes, 16 GCDs | 32 | 64.133 | 60.773 | 3.798 |
