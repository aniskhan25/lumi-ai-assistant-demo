# LUMI vLLM Demo Runbook

Minimal runbook to launch vLLM on LUMI (single-node or multi-node), query it, and benchmark it.

## 1) Prerequisites
- LUMI GPU project/account access
- Working container path in launcher scripts
- Hugging Face access for models that are not already cached

Recommended once per shell:
```bash
module use /appl/local/csc/modulefiles/
module load pytorch
```

Notes:
- The multi-node launcher loads both `pytorch` and `lumi-aif-singularity-bindings`.
- Run commands from repo root: `/scratch/project_462000131/<user>/lumi-ai-assistant-demo`.
- Benchmark outputs under `benchmarks/results/<bench_profile>/job_<jobid>/` are generated artifacts (ignored by git).
- Launch scripts default to `#SBATCH --partition=dev-g`.
- Models are cached under `/scratch/$SLURM_JOB_ACCOUNT/hf-cache`; first use may download the model, later runs reuse the cache.

## 2) Single-Node vLLM
1. Edit `run_vllm_demo.sh`:
- `CONTAINER`
- `MISTRAL_MODEL_DEFAULT` / `QWEN_MODEL_DEFAULT` (or pass `MODEL=...`)
- `#SBATCH --account` (and other Slurm settings if needed)

2. Submit:
```bash
sbatch run_vllm_demo.sh
```

Default model behavior:
```bash
# 1 GPU      -> Mistral default
# multi-GPU -> Qwen default
# MODEL=... -> explicit override
```

3. Save job id:
```bash
JOBID=<jobid>
```

4. Query:
```bash
srun --jobid "$JOBID" --overlap --export=ALL \
  python demo_agent.py \
  --base-url http://127.0.0.1:8000/v1 \
  --question "How do I request 1 GPU on LUMI?"
```

Single-node large-model example:
```bash
sbatch --nodes=1 --gpus-per-node=4 run_vllm_demo.sh
```

## 3) Multi-Node vLLM
1. Edit `run_vllm_demo_multinode.sh`:
- `CONTAINER`
- `MODEL` if you want a different Hugging Face model ID or local model path
- `#SBATCH --account`

2. Submit (example: 2 full GPU nodes):
```bash
sbatch run_vllm_demo_multinode.sh
```

The multi-node launcher follows the LUMI reference and starts vLLM on a Unix socket. Query/chat support for this mode will be added separately after startup is confirmed.

Defaults used by the multi-node launcher:
```bash
MODEL=deepseek-ai/DeepSeek-R1-0528
--tensor-parallel-size $SLURM_GPUS_ON_NODE
--pipeline-parallel-size $SLURM_NNODES
SOCKET_FILE=/tmp/vllm-$SLURM_JOB_ACCOUNT.sock
MASTER_PORT=9999
--enable-expert-parallel
--all2all-backend deepep_low_latency
SBATCH --nodes=2
SBATCH --gpus-per-node=8
SBATCH --cpus-per-task=56
SBATCH --mem=460G
```

3. Save job id:
```bash
JOBID=<jobid>
```

4. Watch startup:
```bash
tail -f demo-mn-${JOBID}.out
```

Expected successful startup progresses past NCCL initialization and reaches model loading:
```bash
vLLM is using nccl==...
rank ... is assigned as ...
Starting to load model ...
```

## 4) Benchmarks
### Single-node
```bash
JOBID=<jobid>
BENCH_PROFILE=small PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 40 4 128
for c in 1 2 4 8 16 32; do
  BENCH_PROFILE=small PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 120 "$c" 128
done
```

Multi-node benchmarking will be added after the reference-style UDS startup is confirmed.

If startup is slow, increase benchmark readiness wait:
```bash
STARTUP_WAIT_S=600 STARTUP_POLL_S=2 \
  BENCH_PROFILE=small PYTHON_BIN=python benchmarks/run_benchmark.sh "$JOBID" 40 4 128
```

### Summarize
```bash
python benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile small
python benchmarks/summarize_results.py --job-id "$JOBID" --bench-profile large
```

## 5) Logs and Debugging
Slurm logs:
```bash
demo-<jobid>.out
demo-<jobid>.err
demo-mn-<jobid>.out
demo-mn-<jobid>.err
```

Runtime logs:
```bash
# Single-node
/scratch/project_462000131/<user>/vllm_runtime/<jobid>/vllm_server.log
```

## 6) LUMI Tuning Knobs
Most runs should use the launcher defaults. Override only when needed:
```bash
MODEL=/path/to/model
EXTRA_VLLM_ARGS="--max-model-len 32768 --max-num-seqs 128 --gpu-memory-utilization 0.9"
```

Notes:
- Single-node defaults to `TP_SIZE=$SLURM_GPUS_ON_NODE`.
- Multi-node follows the LUMI reference launcher: tensor parallelism uses all GPUs on each node and pipeline parallelism uses all nodes.
- Multi-node defaults to expert parallel with `--all2all-backend deepep_low_latency`.
- Use `EXTRA_VLLM_ARGS` for optional vLLM flags such as `--load-format runai_streamer`, `--max-model-len`, `--language-model-only`, or `--trust-remote-code`.

## 7) Benchmark Results and Analysis
The following results use `max_tokens=128` and had `0` failed requests in all shown runs.

### One Node
```text
file,requests,concurrency,max_tokens,ok,failed,p50_s,p95_s,throughput_req_s,throughput_tokens_s,throughput_completion_tokens_s
summary_r120_c32_t128.json,120,32,128,120,0,4.027,4.062,7.443,1138.549,952.668
summary_r120_c16_t128.json,120,16,128,120,0,4.004,4.856,3.649,558.175,467.046
summary_r120_c8_t128.json,120,8,128,120,0,3.991,4.016,2.004,306.528,256.484
summary_r120_c4_t128.json,120,4,128,120,0,3.226,3.249,1.239,189.531,158.588
summary_r40_c4_t128.json,40,4,128,40,0,3.287,4.411,1.176,179.744,150.522
summary_r120_c2_t128.json,120,2,128,120,0,3.212,3.230,0.623,95.264,79.711
summary_r120_c1_t128.json,120,1,128,120,0,3.169,3.197,0.316,48.274,40.393
```

Best one-node profile:
```text
concurrency=32, p95=4.062s, throughput_completion_tokens_s=952.668
```

### Two Node
```text
file,requests,concurrency,max_tokens,ok,failed,p50_s,p95_s,throughput_req_s,throughput_tokens_s,throughput_completion_tokens_s
summary_r120_c128_t128.json,120,128,128,120,0,21.209,21.393,5.601,856.772,716.893
summary_r120_c64_t128.json,120,64,128,120,0,21.427,21.651,2.795,427.516,357.719
summary_r120_c32_t128.json,120,32,128,120,0,20.542,20.985,1.447,221.384,185.240
summary_r120_c16_t128.json,120,16,128,120,0,20.482,26.727,0.710,108.590,90.861
summary_r120_c8_t128.json,120,8,128,120,0,20.184,20.655,0.395,60.468,50.596
summary_r40_c4_t128.json,40,4,128,40,0,16.996,24.077,0.226,34.503,28.894
```

Best two-node profile:
```text
concurrency=128, p95=21.393s, throughput_completion_tokens_s=716.893
```

### Interpretation
- One-node currently outperforms two-node for this setup.
- Best one-node completion throughput (`952.668`) is about `1.33x` the best two-node throughput (`716.893`).
- One-node p95 latency (`4.062s`) is about `5.3x` lower than two-node p95 latency (`21.393s`).
- This indicates distributed overhead dominates for the current model/runtime configuration.
- Current recommendation: use one-node for this model unless you need multi-node for memory/model-size reasons.
