#!/bin/bash
#SBATCH --job-name=lumi-vllm-bench-mn
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --output=bench-mn-%j.out
#SBATCH --error=bench-mn-%j.err

# Serve a model multi-node and benchmark it in the SAME job.
#
# The existing flow is two steps -- sbatch the server, note the job id, then attach with
# `srun --overlap` from another shell -- which is why the benchmark table has gaps: it
# needs a human present while the job happens to be running. This does both, so a result
# either lands or fails on its own.
#
#   MODE=smoke sbatch run_vllm_bench_multinode.sh     # serve only, confirm health, exit
#   MODE=bench sbatch run_vllm_bench_multinode.sh     # serve, then saturation sweep
#
# Kimi-K2 with expert parallelism, the configuration the benchmark table is missing:
#   MODE=bench MODEL=moonshotai/Kimi-K2-Instruct-0905 TP_SIZE=8 PP_SIZE=4 \
#     EXTRA_VLLM_ARGS="--trust-remote-code --quantization fp8 --kv-cache-dtype fp8 \
#     --enable-expert-parallel --all2all-backend deepep_high_throughput \
#     --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 \
#     --gpu-memory-utilization 0.95" sbatch --nodes=4 run_vllm_bench_multinode.sh
#
# Billed to project_462000131. Override with: sbatch --account=<other> ...

set -euo pipefail

CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
MODE="${MODE:-smoke}"
PORT="${PORT:-8000}"
# Kept strictly below the job walltime, unlike the README recipes where
# STARTUP_TIMEOUT_S=14400 against --time=02:00:00 meant the poll could never fire.
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-5400}"
STARTUP_POLL_S="${STARTUP_POLL_S:-5}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
CONCURRENCIES="${CONCURRENCIES:-8 16 32 64 128}"
BENCH_REQUESTS="${BENCH_REQUESTS:-120}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-128}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-600}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --no-enable-prefix-caching}"

NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}}"
TP_SIZE="${TP_SIZE:-${SLURM_GPUS_ON_NODE}}"
PP_SIZE="${PP_SIZE:-${NNODES}}"

echo "Model: ${MODEL}"
echo "Mode: ${MODE}  nodes=${NNODES}  TP=${TP_SIZE}  PP=${PP_SIZE}"
echo "vLLM args: ${EXTRA_VLLM_ARGS}"

# A stalled LAIFS mount is indistinguishable from slow startup, so refuse to start rather
# than mismeasure. run_vllm_demo_multinode.sh still lacks this guard.
LAIFS_PROBE_TIMEOUT_S="${LAIFS_PROBE_TIMEOUT_S:-300}"
for path in /appl/local/laifs/modules "$(dirname "${CONTAINER}")"; do
  if ! timeout "${LAIFS_PROBE_TIMEOUT_S}" ls "${path}" >/dev/null 2>&1; then
    echo "ERROR: ${path} is unreachable (LAIFS mount stalled). Aborting." >&2
    exit 1
  fi
done

module load Local-LAIF lumi-aif-singularity-bindings

WORKDIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUNTIME_DIR="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_runtime/${SLURM_JOB_ID}"
RESULTS_DIR="${WORKDIR}/benchmarks/results/${BENCH_PROFILE:-mn}/job_${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${RESULTS_DIR}"
echo "Results: ${RESULTS_DIR}"

BIND_ARGS=(--bind "${WORKDIR}:/work" --bind "${RUNTIME_DIR}:/runtime")

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR:-${HEAD_NODE}}"
export MASTER_PORT="${MASTER_PORT:-1${SLURM_JOB_ID: -4}}"
export MODEL PORT TP_SIZE PP_SIZE DISTRIBUTED_EXECUTOR_BACKEND EXTRA_VLLM_ARGS NNODES
export LOG_SUBDIR="server"

srun --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=0 --export=ALL \
  singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
  bash /work/launch_vllm_multinode_rank.sh &
LAUNCH_PID=$!

cleanup() {
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
  # Killing the srun leaves the vLLM workers holding their HBM.
  srun --overlap --ntasks="${NNODES}" --ntasks-per-node=1 \
    bash -c 'pkill -f "vllm serve" 2>/dev/null; pkill -f "VLLM::" 2>/dev/null; exit 0' 2>/dev/null || true
}
trap cleanup EXIT

READY_URL="http://127.0.0.1:${PORT}/v1/models"
SECONDS=0
VERDICT="TIMEOUT"
while [ "${SECONDS}" -lt "${STARTUP_TIMEOUT_S}" ]; do
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then VERDICT="DIED"; break; fi
  if curl -fsS --max-time 5 "${READY_URL}" >/dev/null 2>&1; then VERDICT="READY"; break; fi
  sleep "${STARTUP_POLL_S}"
done
STARTUP_SECONDS="${SECONDS}"
# The number run_vllm_demo_multinode.sh computes and throws away.
echo "startup_verdict=${VERDICT} startup_seconds=${STARTUP_SECONDS}"
printf '{"model":"%s","nodes":%s,"tp":%s,"pp":%s,"verdict":"%s","startup_seconds":%s,"args":"%s"}\n' \
  "${MODEL}" "${NNODES}" "${TP_SIZE}" "${PP_SIZE}" "${VERDICT}" "${STARTUP_SECONDS}" "${EXTRA_VLLM_ARGS}" \
  > "${RESULTS_DIR}/startup.json"

if [ "${VERDICT}" != "READY" ]; then
  echo "Server never became ready (${VERDICT}). Tail of rank logs:" >&2
  for r in $(seq 0 $((NNODES - 1))); do
    echo "--- rank ${r} ---" >&2
    tail -n 40 "${RUNTIME_DIR}/server/vllm_server_rank${r}.log" 2>/dev/null >&2 || true
  done
  exit 1
fi

if [ "${MODE}" = "smoke" ]; then
  echo "Smoke test passed: served in ${STARTUP_SECONDS}s. Querying once to confirm generation."
  curl -fsS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"How do I request 1 GPU on LUMI?\"}],\"max_tokens\":48,\"temperature\":0}" \
    | head -c 600
  echo
  exit 0
fi

echo "=== saturation sweep: concurrencies ${CONCURRENCIES} ==="
for c in ${CONCURRENCIES}; do
  echo "--- concurrency ${c} ---"
  srun --overlap --ntasks=1 --nodes=1 -w "${HEAD_NODE}" \
    singularity run "${BIND_ARGS[@]}" "${CONTAINER}" \
    python3 /work/benchmarks/benchmark_openai.py \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --prompts-file /work/benchmarks/prompts.txt \
      --requests "${BENCH_REQUESTS}" --concurrency "${c}" \
      --max-tokens "${BENCH_MAX_TOKENS}" --timeout "${REQUEST_TIMEOUT_S}" \
      --startup-wait-s 60 \
      --output-json "/work/benchmarks/results/${BENCH_PROFILE:-mn}/job_${SLURM_JOB_ID}/summary_r${BENCH_REQUESTS}_c${c}_t${BENCH_MAX_TOKENS}.json" \
    || echo "  concurrency ${c} failed"
done

echo "=== summary ==="
echo "startup_seconds=${STARTUP_SECONDS}"
grep -h throughput_completion_tokens_s "${RESULTS_DIR}"/summary_*.json 2>/dev/null || true
