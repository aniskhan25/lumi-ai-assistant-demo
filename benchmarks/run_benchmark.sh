#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <jobid> [requests] [concurrency] [max_tokens]" >&2
  exit 2
fi

JOBID="$1"
REQUESTS="${2:-40}"
CONCURRENCY="${3:-4}"
MAX_TOKENS="${4:-128}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SRUN_NODELIST="${SRUN_NODELIST:-}"
STARTUP_WAIT_S="${STARTUP_WAIT_S:-180}"
STARTUP_POLL_S="${STARTUP_POLL_S:-2}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-120}"
BENCH_PROFILE="${BENCH_PROFILE:-default}"  # e.g. small, large

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/results/${BENCH_PROFILE}/job_${JOBID}"

mkdir -p "${OUT_DIR}"

echo "Running benchmark on job ${JOBID}"
echo "Base URL: ${BASE_URL}"
echo "Benchmark profile: ${BENCH_PROFILE}"
echo "Requests=${REQUESTS}, Concurrency=${CONCURRENCY}, MaxTokens=${MAX_TOKENS}"
echo "Startup wait: ${STARTUP_WAIT_S}s (poll ${STARTUP_POLL_S}s)"
echo "Request timeout: ${REQUEST_TIMEOUT_S}s"
if [ -n "${SRUN_NODELIST}" ]; then
  echo "Pinned to node: ${SRUN_NODELIST}"
fi

srun_args=(--jobid "${JOBID}" --overlap)
if [ -n "${SRUN_NODELIST}" ]; then
  srun_args+=(-w "${SRUN_NODELIST}")
fi

srun "${srun_args[@]}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_openai.py" \
  --base-url "${BASE_URL}" \
  --prompts-file "${REPO_ROOT}/benchmarks/prompts.txt" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --timeout "${REQUEST_TIMEOUT_S}" \
  --startup-wait-s "${STARTUP_WAIT_S}" \
  --startup-poll-s "${STARTUP_POLL_S}" \
  --output-json "${OUT_DIR}/summary_r${REQUESTS}_c${CONCURRENCY}_t${MAX_TOKENS}.json"
