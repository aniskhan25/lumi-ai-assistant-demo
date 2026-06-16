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
BENCH_PROFILE=small benchmarks/run_saturation.sh "$JOBID" 120 128 "1 2 4 8 16 32"
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

## Known Working Multi-Node Launches

DeepSeek-R1:

```bash
MODEL=deepseek-ai/DeepSeek-R1-0528 \
TP_SIZE=8 \
PP_SIZE=2 \
EXTRA_VLLM_ARGS="--enable-expert-parallel --all2all-backend deepep_low_latency" \
sbatch run_vllm_demo_multinode.sh
```

GPT-OSS 120B:

```bash
MODEL=openai/gpt-oss-120b \
TP_SIZE=8 \
PP_SIZE=2 \
EXTRA_VLLM_ARGS="--max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching" \
sbatch run_vllm_demo_multinode.sh
```

Kimi-K2:

```bash
MODEL=moonshotai/Kimi-K2-Instruct-0905 \
TP_SIZE=8 \
PP_SIZE=2 \
EXTRA_VLLM_ARGS="--trust-remote-code --load-format runai_streamer --quantization fp8 --kv-cache-dtype fp8 --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 512 --gpu-memory-utilization 0.98" \
sbatch run_vllm_demo_multinode.sh
```

The Kimi-K2 launch starts, but `--max-num-seqs 1` limits throughput.

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
