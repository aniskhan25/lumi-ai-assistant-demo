#!/bin/bash
#SBATCH --job-name=dsv4-serve
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --output=dsv4-serve-%j.out

# Phase 0 for DeepSeek-V4.1-Flash, end to end: serve DeepSeek-V4-Flash, which LAIF's
# vLLM 0.26 supports natively and which shares V4.1's sparse-MLA attention, indexer and
# compressor design, on one LUMI-G node with the stock container, no plugin. Then the
# smoke and output-correctness probes. A Lexivo report (Oct 2026) had V4-Flash starting
# on vLLM 0.22.1 and then producing NaN logits.
#
#   sbatch repro/deepseek_v41_gfx90a/run_v4_serve.sh              # from the repo root
#   EXTRA_VLLM_ARGS="..." sbatch ...                               # override/extend flags
#   PROBE=0 BENCH=1                                                # speed sweep instead of probe
#   COMPILE=1                                                      # CUDA graphs, not eager
#   COMPILE=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE TRITON_LOGITS=1        # also full decode graphs
#
# Phase-0 workarounds, both off by default (see README):
#   AMDGCN_USE_BUFFER_OPS=0  Triton 3.6's AMD CanonicalizePointers pass asserts on V4's
#                            fused q/kv RMSNorm kernel (pointer chosen inside an if);
#                            this env var skips that pass (job 23491948)
#   INDEXER_GATE=1           lift 0.26's aiter-only gate on the sparse indexer
#                            (indexer_gate/, job 23492281)
#   TORCH_MHC=1              (branch bug/dsv4-flash-gfx90a-garbage) turn off the TileLang
#                            hyper-connection kernels on gfx90a, as 0.26 does on gfx942.
#                            Uses repro/dsv4_flash_debug/plugin, which also carries the
#                            indexer gate (DSV4_INDEXER_GATE).
set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SNAPSHOT="$(ls -d /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/* | head -n1)"
GSM8K="$(ls /scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache/hub/datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet | head -n1)"
PORT="$((20000 + ${SLURM_JOB_ID} % 20000))"   # shared nodes share ports
LOG="${REPO}/benchmarks/results/dsv4-flash/job_${SLURM_JOB_ID}"
mkdir -p "${LOG}"

module load Local-LAIF lumi-aif-singularity-bindings
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
export HF_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache" HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm-cache"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}" MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
# Shared node: a co-tenant's job end wipes /dev/shm (laifs-container-recipes#46).
SHM="/tmp/shm-${SLURM_JOB_ID}"; mkdir -p "${SHM}"
RUN=(singularity exec -B "${SHM}:/dev/shm")
# The container sets PYTHONPATH itself, overriding the host value; pass ours with --env.
if [ "${TORCH_MHC:-0}" = 1 ]; then
  export DSV4_TORCH_MHC=1 DSV4_INDEXER_GATE="${INDEXER_GATE:-0}"
  export DSV4_TRITON_PAGED_LOGITS="${TRITON_LOGITS:-0}"
  RUN+=(--env "PYTHONPATH=${REPO}/repro/dsv4_flash_debug/plugin:/opt/venv/lib/python3.12/site-packages")
elif [ "${INDEXER_GATE:-0}" = 1 ]; then
  RUN+=(--env "PYTHONPATH=${REPO}/repro/deepseek_v41_gfx90a/indexer_gate:/opt/venv/lib/python3.12/site-packages")
fi
RUN+=("${CONTAINER}")
echo "AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS:-unset} INDEXER_GATE=${INDEXER_GATE:-0} TORCH_MHC=${TORCH_MHC:-0} TRITON_LOGITS=${TRITON_LOGITS:-0} COMPILE=${COMPILE:-0} CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-PIECEWISE}"

# MXFP4 experts: on ROCm vLLM's automatic MoE backend choice is aiter only (as for MiMo),
# so ask for the Triton dequantising one. V4's attention stores K/V only as fp8_ds_mla
# (job 23491881: "fp8_ds_mla layout only supports fp8 kv-cache, got auto").
# Eager by default: fewer moving parts. COMPILE=1 drops --enforce-eager (branch
# perf/dsv4-flash-gfx90a-compile, needs TORCH_MHC=1 INDEXER_GATE=1). V4 on 0.26 is not
# torch.compiled; it uses "breakable" graphs. PIECEWISE, because FULL graphs capture the
# indexer, whose torch paged-MQA-logits fallback calls .item() per sequence and cannot be
# captured (job 23512857); in PIECEWISE the indexer and attention run as eager breaks.
# TRITON_LOGITS=1 replaces that fallback with a capturable kernel, so FULL_AND_PIECEWISE
# (vLLM's default: whole decode steps as one graph) can be tried.
if [ "${COMPILE:-0}" = 1 ]; then
  DEFAULT_EXTRA="--moe-backend triton_unfused --compilation-config {\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\"}"
else DEFAULT_EXTRA="--moe-backend triton_unfused --enforce-eager"; fi
SERVE=(vllm serve "${SNAPSHOT}" --served-model-name dsv4 --port "${PORT}"
       --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 32
       --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92
       --no-enable-prefix-caching --kv-cache-dtype fp8
       ${EXTRA_VLLM_ARGS:-${DEFAULT_EXTRA}})
echo "Serving: ${SERVE[*]}"
"${RUN[@]}" "${SERVE[@]}" > "${LOG}/server.log" 2>&1 &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null; wait ${SERVER} 2>/dev/null; pkill -u ${USER} -f "VLLM::" 2>/dev/null || true' EXIT

SECONDS=0
until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER}" 2>/dev/null || [ "${SECONDS}" -gt 3600 ]; then
    echo "SERVER_FAILED after ${SECONDS}s"; grep -v amdsmi "${LOG}/server.log" | tail -80; exit 1
  fi
  sleep 5
done
echo "SERVER_READY after ${SECONDS}s"
grep -hE "Loading weights took|Model loading took|backend|KV cache|Maximum concurrency|compil|[Gg]raph|cudagraph_mode" "${LOG}/server.log" \
  | grep -vE "amdsmi|Worker_TP[1-9]" | head -24 || true   # head closes early: SIGPIPE

STATUS=0
# The Qwen3.8 branch's smoke_requests.py is not on this branch; the probe covers the
# same ground for correctness (non-finite logprobs, garbage, 2.3k/18.2k needle, GSM8K).
if [ "${PROBE:-1}" = 1 ]; then
  "${RUN[@]}" python "${REPO}/repro/lexivo_mimo_gfx90a/probe.py" \
    --base-url "http://127.0.0.1:${PORT}" --model dsv4 \
    --gsm8k-parquet "${GSM8K}" --gsm8k-n "${PROBE_GSM8K_N:-200}" --out "${LOG}/probe.json" \
    || STATUS=1
fi

# BENCH=1: streaming `vllm bench serve` sweep, as for Qwen3.8-Flash (repro/qwen38_flash_gfx90a)
# so the numbers compare. Warm-up first: the first requests of an allocation pay one-time
# kernel compilation. PROMPTS_PER_SLOT is small because eager V4 is slow (~37 tok/s at c=16).
if [ "${BENCH:-0}" = 1 ]; then
  bench() {  # concurrency, prompts, input len, output len, result name
    "${RUN[@]}" vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "http://127.0.0.1:${PORT}" --model dsv4 --tokenizer "${SNAPSHOT}" \
      --tokenizer-mode deepseek_v4 --dataset-name random --random-input-len "$3" \
      --random-output-len "$4" --ignore-eos --num-prompts "$2" --max-concurrency "$1" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-dir "${LOG}" --result-filename "$5.json" \
      | grep -E "Output token throughput|Mean TTFT|Median TPOT|P99 TPOT|Successful|Error" || true
  }
  WORKLOADS="${WORKLOADS:-1024:512}"
  first="${WORKLOADS%% *}"
  echo "=== warm-up ==="; bench 4 4 "${first%%:*}" "${first##*:}" warmup
  for w in ${WORKLOADS}; do
    for c in ${CONCURRENCIES:-1 4 16 32}; do
      n=$((c * ${PROMPTS_PER_SLOT:-2})); [ "${n}" -lt 4 ] && n=4
      echo "=== bench c=${c} n=${n} in:out=${w} ==="
      bench "${c}" "${n}" "${w%%:*}" "${w##*:}" "serve_c${c}_i${w%%:*}_o${w##*:}"
    done
  done
fi
[ "${STATUS}" != 0 ] && { echo "--- server log tail ---"; grep -v amdsmi "${LOG}/server.log" | tail -60; }
exit "${STATUS}"
