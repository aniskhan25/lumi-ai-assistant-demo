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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/results/job_${JOBID}"

mkdir -p "${OUT_DIR}"

echo "Running benchmark on job ${JOBID}"
echo "Base URL: ${BASE_URL}"
echo "Requests=${REQUESTS}, Concurrency=${CONCURRENCY}, MaxTokens=${MAX_TOKENS}"

srun --jobid "${JOBID}" --overlap \
  python3 "${REPO_ROOT}/benchmarks/benchmark_openai.py" \
  --base-url "${BASE_URL}" \
  --prompts-file "${REPO_ROOT}/benchmarks/prompts_puhti.txt" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --output-json "${OUT_DIR}/summary_r${REQUESTS}_c${CONCURRENCY}_t${MAX_TOKENS}.json" \
  --output-raw-json "${OUT_DIR}/raw_r${REQUESTS}_c${CONCURRENCY}_t${MAX_TOKENS}.json"
