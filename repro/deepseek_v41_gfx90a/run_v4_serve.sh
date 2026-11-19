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
#   MORE_VLLM_ARGS="..."                                           # append flags, keep defaults
#   PROBE=0 BENCH=1                                                # speed sweep instead of probe
#   COMPILE=1                                                      # CUDA graphs, not eager
#   COMPILE=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE TRITON_LOGITS=1        # also full decode graphs
#   NO_ASYNC=1                                                     # --no-async-scheduling
#   NUMA_BIND=1                                                    # pin workers to their GCD's CCD
#   BF16_LINEAR=1                                                  # block-FP8 dense linears in bf16
#     (activations still FP8-rounded; BF16_ACT=bf16 skips that and costs accuracy)
#   FP8_ROUND=torch                                                # vLLM quantiser + dequant instead of the fused kernel
#   PROFILE=1 PROBE=0                                              # torch-profile decode at
#     c=1 and c=32 (eager: kernels in graphs can't be attributed); rank-0 component split
#     by repro/dsv4_flash_debug/analyze_trace.py
#     (with COMPILE=1 also graph_trace.py: idle and host-late gaps per step; STEP_MOE=43
#      without MTP)
#     (needed with TRITON_LOGITS=1: with async scheduling the server hangs after a few
#      decode steps, jobs 23518755, 23519175; the torch fallback's per-sequence syncs hid it)
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
#   MHC_FIX=triton|wave64    instead of TORCH_MHC: keep the fused TileLang mHC path and
#                            replace only its wrong pre "big fuse" kernel (plugin mhc_fix.py)
#   MHC_CHECK=1              (eager) log mHC calls that are non-finite or differ from the reference
#   MHC_NATIVE=head,fusedpost,fused,pre   route those mHC ops to reference kernels (bisection)
#   MHC_META=1 / MHC_NANTRACE=1   log mHC argument layouts / where a NaN first appears (graph-safe)
#   ROUTER_GUARD=1|0|count   nan_to_num MoE gate logits before ROCm top-k (default 1; plugin router_guard.py)
#   GRAPH_DIFF=1             rerun each graph step eagerly, log first diverging layer site
#   SPARSE_SPLIT=1           split-K sparse attention decode (vLLM gates it to gfx942/950; plugin sparse_split.py)
#   SPARSE_SPLIT_MAX=16      ... for decode batches of at most this many queries
#   SPARSE_DECODE=1          sparse attention decode with the V4.1 split-K kernel, all batch
#                            sizes (plugin sparse_decode.py); replaces SPARSE_SPLIT
#   WO_A=1                   attention wo_a GEMM via wvSplitK / Triton, not einsum (wo_a.py)
#   CUSTOM_AR=copy|fence     vLLM custom all-reduce instead of RCCL (vLLM gates it to gfx94x/95x; 1 = copy)
#   SCORE_GEMM=1             compressor score GEMMs in Triton instead of hipBLASLt's fp32-MFMA kernel (plugin score_gemm.py)
#   MOE_SMALL=1              MoE at decode sizes: pack_bitmatrix row block, expert GEMM tiles (plugin moe_small.py)
#   GATE_GEMM=1              MoE router GEMM in Triton with fp32 logits (plugin moe_small.py)
#   MTP=N                    speculative decoding with V4's MTP layer, N draft tokens (1 is best above 1 user;
#                            needs TRITON_LOGITS=1 for the indexer at next_n = N + 1)
#   NANFILL=1                fill fresh float allocations with NaN (exposes reads of unwritten memory)
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
PLUGIN=0; { [ "${TORCH_MHC:-0}" = 1 ] || [ -n "${MHC_FIX:-}" ]; } && PLUGIN=1
if [ "${PLUGIN}" = 1 ]; then
  export DSV4_TORCH_MHC="${TORCH_MHC:-0}" DSV4_MHC_FIX="${MHC_FIX:-}" DSV4_MHC_CHECK="${MHC_CHECK:-0}" DSV4_MHC_NATIVE="${MHC_NATIVE:-}" DSV4_MHC_META="${MHC_META:-0}" DSV4_MHC_NANTRACE="${MHC_NANTRACE:-0}" DSV4_NANFILL="${NANFILL:-0}" DSV4_STACK_DIR="${LOG}/stacks" DSV4_INDEXER_GATE="${INDEXER_GATE:-0}"
  export DSV4_TRITON_PAGED_LOGITS="${TRITON_LOGITS:-0}" DSV4_PAGED_LOGITS_CHECK="${LOGITS_CHECK:-0}"
  export DSV4_BF16_LINEAR="${BF16_LINEAR:-0}" DSV4_BF16_LINEAR_ACT="${BF16_ACT:-fp8}" DSV4_FP8_ROUND="${FP8_ROUND:-triton}" DSV4_ROUTER_GUARD="${ROUTER_GUARD:-1}" DSV4_GRAPH_DIFF="${GRAPH_DIFF:-0}" DSV4_SPARSE_SPLIT="${SPARSE_SPLIT:-0}" DSV4_SPARSE_SPLIT_MAX="${SPARSE_SPLIT_MAX:-16}" DSV4_CUSTOM_AR="${CUSTOM_AR:-0}" DSV4_SCORE_GEMM="${SCORE_GEMM:-0}" DSV4_MOE_SMALL="${MOE_SMALL:-0}" DSV4_GATE_GEMM="${GATE_GEMM:-0}" DSV4_SPARSE_DECODE="${SPARSE_DECODE:-0}" DSV4_WO_A="${WO_A:-0}"
  RUN+=(--env "PYTHONPATH=${REPO}/repro/dsv4_flash_debug/plugin:/opt/venv/lib/python3.12/site-packages")
elif [ "${INDEXER_GATE:-0}" = 1 ]; then
  RUN+=(--env "PYTHONPATH=${REPO}/repro/deepseek_v41_gfx90a/indexer_gate:/opt/venv/lib/python3.12/site-packages")
fi
RUN+=("${CONTAINER}")
echo "AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS:-unset} INDEXER_GATE=${INDEXER_GATE:-0} TORCH_MHC=${TORCH_MHC:-0} MHC_FIX=${MHC_FIX:-} MHC_NATIVE=${MHC_NATIVE:-} MHC_CHECK=${MHC_CHECK:-0} TRITON_LOGITS=${TRITON_LOGITS:-0} NO_ASYNC=${NO_ASYNC:-0} BF16_LINEAR=${BF16_LINEAR:-0} BF16_ACT=${BF16_ACT:-fp8} COMPILE=${COMPILE:-0} ROUTER_GUARD=${ROUTER_GUARD:-1} GRAPH_DIFF=${GRAPH_DIFF:-0} SPARSE_SPLIT=${SPARSE_SPLIT:-0}/${SPARSE_SPLIT_MAX:-16} SPARSE_DECODE=${SPARSE_DECODE:-0} WO_A=${WO_A:-0} CUSTOM_AR=${CUSTOM_AR:-0} SCORE_GEMM=${SCORE_GEMM:-0} MOE_SMALL=${MOE_SMALL:-0} GATE_GEMM=${GATE_GEMM:-0} MTP=${MTP:-0} NUMA_BIND=${NUMA_BIND:-0} CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-PIECEWISE}"

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
[ "${NO_ASYNC:-0}" = 1 ] && DEFAULT_EXTRA="${DEFAULT_EXTRA} --no-async-scheduling"
[ "${MTP:-0}" != 0 ] && DEFAULT_EXTRA="${DEFAULT_EXTRA} --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"
# NUMA_BIND=1: pin each worker to its GCD's CCD (LUMI-G map; core 0 of each CCD is reserved)
# and NUMA node. Graph-mode profiles showed a random rank's host launching late.
[ "${NUMA_BIND:-0}" = 1 ] && DEFAULT_EXTRA="${DEFAULT_EXTRA} --numa-bind --numa-bind-nodes 3 3 1 1 0 0 2 2 --numa-bind-cpus 49-55 57-63 17-23 25-31 1-7 9-15 33-39 41-47"
PROF_DIR="${LOG}/profile"
[ "${PROFILE:-0}" = 1 ] && DEFAULT_EXTRA="${DEFAULT_EXTRA} --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROF_DIR}\",\"ignore_frontend\":true}"
SERVE=(vllm serve "${SNAPSHOT}" --served-model-name dsv4 --port "${PORT}"
       --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 32
       --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92
       --no-enable-prefix-caching --kv-cache-dtype fp8
       ${EXTRA_VLLM_ARGS:-${DEFAULT_EXTRA}} ${MORE_VLLM_ARGS:-})
echo "Serving: ${SERVE[*]}"
echo "CPUs allowed: $(taskset -cp $$ | cut -d: -f2) ($(nproc))"
"${RUN[@]}" "${SERVE[@]}" > "${LOG}/server.log" 2>&1 &
SERVER=$!
trap 'kill ${WATCH:-} 2>/dev/null || true; kill ${SERVER} 2>/dev/null; wait ${SERVER} 2>/dev/null; pkill -u ${USER} -f "VLLM::" 2>/dev/null || true' EXIT

# Hang watch: when vLLM says the workers stopped answering, dump their Python stacks
# (the plugin registers faulthandler on SIGUSR1).
if [ "${PLUGIN}" = 1 ]; then
  # vLLM also prints the stall message during startup warm-up: count only later ones.
  ( until grep -q "Application startup complete" "${LOG}/server.log" 2>/dev/null; do sleep 10; done
    stalls() { grep -c "No available shared memory broadcast block" "${LOG}/server.log" || true; }
    base=$(stalls)
    while sleep 20; do
      if [ "$(stalls)" -gt "${base}" ]; then
        D="${LOG}/stacks"; mkdir -p "${D}"
        echo "HANG $(date +%T): signalling $(pgrep -u "${USER}" -f "VLLM::" | wc -l) vLLM processes; dumps in ${D}"
        for k in 1 2; do "${RUN[@]}" rocm-smi --showuse --showmemuse > "${D}/gpu-use-${k}.txt" 2>&1; sleep 5; done
        pkill -USR1 -u "${USER}" -f "VLLM::" || true; sleep 10
        # Native stacks: a process stuck in a GPU sync or an RCCL call shows it here.
        for pid in $(pgrep -u "${USER}" -f "VLLM::(Worker_TP0|EngineCore)"); do
          timeout 180 "${RUN[@]}" rocgdb -p "${pid}" -batch -ex "thread apply all bt 25" \
            > "${D}/gdb-${pid}.txt" 2>&1 || true
        done
        echo "HANG: dumps written; stopping the server"; kill "${SERVER}" 2>/dev/null || true; break
      fi
    done ) &
  WATCH=$!
fi
SECONDS=0
until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER}" 2>/dev/null || [ "${SECONDS}" -gt 3600 ]; then
    echo "SERVER_FAILED after ${SECONDS}s"; grep -v amdsmi "${LOG}/server.log" | tail -80; exit 1
  fi
  sleep 5
done
echo "SERVER_READY after ${SECONDS}s"
grep -hE "Loading weights took|Model loading took|backend|KV cache|Maximum concurrency|compil|[Gg]raph|cudagraph_mode|DSV4_BF16_LINEAR" "${LOG}/server.log" \
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
if [ "${PROFILE:-0}" = 1 ]; then
  for c in ${PROFILE_CONCURRENCIES:-1 32}; do
    echo "=== profile c=${c} ==="
    "${RUN[@]}" python "${REPO}/repro/dsv4_flash_debug/profile_decode.py" \
      "http://127.0.0.1:${PORT}" dsv4 "${c}" "${PROFILE_SECONDS:-5}" || STATUS=1
    sleep 20   # traces are written after /stop_profile returns
    mkdir -p "${PROF_DIR}/c${c}"
    find "${PROF_DIR}" -maxdepth 1 -type f -exec mv {} "${PROF_DIR}/c${c}/" \;
    ls -la "${PROF_DIR}/c${c}" | head
    rank0="$(ls "${PROF_DIR}/c${c}"/*rank0* "${PROF_DIR}/c${c}"/*.json.gz 2>/dev/null | head -1)"
    [ -n "${rank0}" ] && "${RUN[@]}" python "${REPO}/repro/dsv4_flash_debug/analyze_trace.py" "${rank0}"
    # Graph mode: idle time and host-late gaps per step (MoE layers per step: 43, +1 with MTP)
    [ -n "${rank0}" ] && [ "${COMPILE:-0}" = 1 ] && "${RUN[@]}" python \
      "${REPO}/repro/dsv4_flash_debug/graph_trace.py" "${rank0}" pack_bitmatrix "${STEP_MOE:-44}"
  done
fi
[ "${STATUS}" != 0 ] && { echo "--- server log tail ---"; grep -v amdsmi "${LOG}/server.log" | tail -60; }
exit "${STATUS}"
