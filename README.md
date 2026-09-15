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

```bash
MODEL=openai/gpt-oss-120b \
TP_SIZE=8 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching" \
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

Two full nodes (avoid PP bubble):

```bash
STARTUP_TIMEOUT_S=2700 \
MODEL=deepseek-ai/DeepSeek-R1-0528 \
TP_SIZE=16 \
PP_SIZE=1 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--enable-expert-parallel --all2all-backend deepep_high_throughput --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95" \
sbatch run_vllm_demo_multinode.sh
```

Two full nodes (larger micro-batches):

```bash
STARTUP_TIMEOUT_S=2700 \
MODEL=deepseek-ai/DeepSeek-R1-0528 \
TP_SIZE=8 \
PP_SIZE=2 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--enable-expert-parallel --all2all-backend deepep_high_throughput --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.95" \
sbatch run_vllm_demo_multinode.sh
```

Two full nodes

```bash
STARTUP_TIMEOUT_S=3600 \
MODEL=meta-llama/Llama-3.1-405B-Instruct \
TP_SIZE=8 \
PP_SIZE=2 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching" \
sbatch run_vllm_demo_multinode.sh
```

Four full nodes, Kimi-K2 — best configuration measured (2.928 tok/s per GCD, ~31 min
startup). Expert parallelism for the startup win, `--max-num-seqs 64` for the throughput:

```bash
MODE=bench BENCH_PROFILE=kimi MODEL=moonshotai/Kimi-K2-Instruct-0905 \
TP_SIZE=8 PP_SIZE=4 STARTUP_TIMEOUT_S=5400 CONCURRENCIES="64 128" \
EXTRA_VLLM_ARGS="--trust-remote-code --quantization fp8 --kv-cache-dtype fp8 \
  --enable-expert-parallel --all2all-backend deepep_high_throughput \
  --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95" \
sbatch --nodes=4 --time=03:00:00 run_vllm_bench_multinode.sh
```

Four full nodes, original recipe (kept for comparison; 1.961 tok/s per GCD, ~2 h startup):

```bash
STARTUP_TIMEOUT_S=14400 \
MODEL=moonshotai/Kimi-K2-Instruct-0905 \
TP_SIZE=8 \
PP_SIZE=4 \
RUNAI_STREAMER_CONCURRENCY=4 \
EXTRA_VLLM_ARGS="--trust-remote-code --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95" \
sbatch --nodes=4 --time=02:00:00 run_vllm_demo_multinode.sh
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
| Single node | `Qwen/Qwen2.5-32B-Instruct` | 2 GCDs | 64 | 26.397 | 293.506 | 146.753 |
|  | `Qwen/Qwen2.5-72B-Instruct` | 4 GCDs | 64 | 3.336 | 2319.813 | 579.953 |
|  | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 8 GCDs | 96 | 25.226 | 455.677 | 56.960 |
|  | `openai/gpt-oss-120b` | 8 GCDs | 128 | 19.365 | 791.917 | 98.990 |
| Multi-node | `deepseek-ai/DeepSeek-R1-0528` | 2 nodes, 16 GCDs | 32 | 64.133 | 60.773 | 3.798 |
|  | `deepseek-ai/DeepSeek-R1-0528` (avoid PP bubble) | 2 nodes, 16 GCDs | 64 | 103.529 | 74.789 | 4.674 |
|  | `deepseek-ai/DeepSeek-R1-0528` (larger micro-batches) | 2 nodes, 16 GCDs | 64 | 70.551 | 110.600 | 6.912 |
|  | `meta-llama/Llama-3.1-405B-Instruct` | 2 nodes, 16 GCDs | 32 | 34.373 | 113.057 | 7.066 |
|  | `meta-llama/Llama-3.1-405B-Instruct` | 4 nodes, 32 GCDs | 160 | 66.241 | 227.973 | 7.124 |
|  | `moonshotai/Kimi-K2-Instruct-0905` | 4 nodes, 32 GCDs | 64 | 121.763 | 62.749 | 1.961 |
|  | `moonshotai/Kimi-K2-Instruct-0905` (re-measured, job 22028722) | 4 nodes, 32 GCDs | 32 | 62.142 | 61.620 | 1.926 |
|  | `moonshotai/Kimi-K2-Instruct-0905` (expert parallel, job 22028721) | 4 nodes, 32 GCDs | 64 | 139.097 | 54.403 | 1.700 |
|  | `moonshotai/Kimi-K2-Instruct-0905` (expert parallel, `--max-num-seqs 64`, job 22060642) | 4 nodes, 32 GCDs | 128 | 156.561 | 93.695 | **2.928** |
|  | `moonshotai/Kimi-K2-Instruct-0905` (expert parallel, `--max-num-seqs 128`, job 22065599) | 4 nodes, 32 GCDs | 256 | 189.897 | 77.048 | 2.408 |

### Expert parallelism for Kimi-K2: a startup win, not a throughput win

Measured head to head on 4 nodes, same day, same harness (jobs 22028721 / 22028722):

| | startup | best concurrency | p95 | tok/s per GCD |
| --- | ---: | ---: | ---: | ---: |
| no expert parallelism | **7603 s** | 32 | 62.1 s | **1.926** |
| `--enable-expert-parallel --all2all-backend deepep_high_throughput` | **1547 s** | 64 | 139.1 s | 1.700 |

Best configuration found so far: **expert parallelism plus `--max-num-seqs 64`**, at
2.928 tok/s per GCD and a ~31 minute startup.

Expert parallelism makes startup **5x faster** — each rank loads only its own experts
rather than the full expert set for TP sharding, which matters a lot for a 958 GiB
checkpoint on Lustre (weight loading alone took 6459-7117 s without it). It costs about
12% of throughput, so enable it when iteration speed or queue time matters and leave it
off when steady-state throughput is the goal.

The re-measured no-EP row reproduces the original 1.961 to within 2%, so the original
measurement was sound.

**The throughput ceiling was `--max-num-seqs`, not the parallelism strategy.** Both
configurations above saturate at concurrency 32 — flat throughput from 32 to 128 while
p95 quadruples — because the recipe sets `--max-num-seqs 32`. Raising it to 64, changing
nothing else, gives **1.72x** the throughput (54.4 -> 93.7 tok/s) and beats the original
row by 1.5x:

| `--max-num-seqs` | best concurrency | p95 | tok/s | per GCD |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 139.1 s | 54.403 | 1.700 |
| **64** | **128** | **156.6 s** | **93.695** | **2.928** |
| 128 | 256 | 189.9 s | 77.048 | 2.408 |

**64 is an optimum, not a floor.** Going to 128 costs 18% of throughput and pushes p95 to
190 s. It is not a memory limit — rank 0 reports 28.77 GiB / 2,914,560 KV tokens and
`Maximum concurrency for 16,384 tokens per request: 177.89x`, so 128 sequences fit
comfortably. Admitting more sequences than the batched-token budget can feed just trades
throughput for queueing, most likely prefill/decode interference at a fixed
`--max-num-batched-tokens 8192`. The untested middle is 96; the useful pairing to try
next is a higher `--max-num-seqs` **with** a higher `--max-num-batched-tokens`, which
needs care because that combination lost a worker (see below).

What does bind is **activation memory**: a run at `--max-num-seqs 128
--max-num-batched-tokens 16384` lost a worker during startup before KV sizing (job
22032772). Since the KV figures show 128 sequences fit, the batched-token increase is the
likely cause — raise `--max-num-seqs` alone and leave `--max-num-batched-tokens` at 8192.

## Known Issues

**Fixed in the launchers.** Multi-node RCCL hung indefinitely while creating additional
communicators, which is why the multi-node recipes above carry `STARTUP_TIMEOUT_S` of
2700-14400 s. Cause: libfabric's vendor-default MR cache monitor (`memhooks`) does not
see ROCm memory remapping, so a stale registration makes an RDMA silently never complete.
The launchers now set `FI_MR_CACHE_MONITOR=userfaultfd` — 0 hangs in 18 attempts against
21/23 without it, at no bandwidth cost. Tracked upstream as
[laifs-container-recipes#44](https://github.com/lumi-ai-factory/laifs-container-recipes/issues/44);
the fix here is a workaround, so keep generous startup timeouts until that closes.

Also do **not** set `NCCL_NET_GDR_LEVEL=PHB`: harmless on one node, hangs every rank from
two nodes upward ([#30](https://github.com/lumi-ai-factory/laifs-container-recipes/issues/30)).

`repro/rccl_startup_gfx90a/` times RCCL startup phase by phase with no model in the
picture, then checks whether the same signature appears in real vLLM startup, and
measures what capping channels costs in collective bandwidth:

```bash
sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh              # cheapest, no model
MODE=sweep sbatch repro/rccl_startup_gfx90a/run_rccl_probe.sh   # full variant table
sbatch repro/rccl_startup_gfx90a/run_bandwidth.sh               # cost of capping channels
```

See `repro/rccl_startup_gfx90a/README.md` for the run order and
`repro/rccl_startup_gfx90a/FINDINGS.md` for conclusions.
