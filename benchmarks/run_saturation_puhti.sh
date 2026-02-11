#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <jobid> [requests] [max_tokens] [concurrency_list]" >&2
  echo "Example: $0 31752419 120 128 \"8 10 12 16\"" >&2
  exit 2
fi

JOBID="$1"
REQUESTS="${2:-120}"
MAX_TOKENS="${3:-128}"
CONCURRENCY_LIST="${4:-8 10 12 16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for c in ${CONCURRENCY_LIST}; do
  echo
  echo "=== Saturation run: concurrency=${c}, requests=${REQUESTS}, max_tokens=${MAX_TOKENS} ==="
  "${SCRIPT_DIR}/run_benchmark_puhti.sh" "${JOBID}" "${REQUESTS}" "${c}" "${MAX_TOKENS}"
done

