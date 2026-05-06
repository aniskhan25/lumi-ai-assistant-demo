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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/results/job_${JOBID}"

ensure_python_bin() {
  local bin="$1"
  if command -v "${bin}" >/dev/null 2>&1; then
    echo "${bin}"
    return 0
  fi

  if ! type module >/dev/null 2>&1; then
    [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh || true
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash || true
  fi
  if type module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles/ >/dev/null 2>&1 || true
    module load pytorch >/dev/null 2>&1 || true
  fi

  if command -v "${bin}" >/dev/null 2>&1; then
    echo "${bin}"
    return 0
  fi
  if [ "${bin}" != "python" ] && command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if [ "${bin}" != "python3" ] && command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  return 1
}

PYTHON_BIN="$(ensure_python_bin "${PYTHON_BIN}")" || {
  echo "Could not find a usable python binary for benchmarks." >&2
  echo "Tried module auto-load: module use /appl/local/csc/modulefiles/ ; module load pytorch" >&2
  exit 127
}

mkdir -p "${OUT_DIR}"

echo "Running benchmark on job ${JOBID}"
echo "Base URL: ${BASE_URL}"
echo "Requests=${REQUESTS}, Concurrency=${CONCURRENCY}, MaxTokens=${MAX_TOKENS}"
echo "Startup wait: ${STARTUP_WAIT_S}s (poll ${STARTUP_POLL_S}s)"
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
  --startup-wait-s "${STARTUP_WAIT_S}" \
  --startup-poll-s "${STARTUP_POLL_S}" \
  --output-json "${OUT_DIR}/summary_r${REQUESTS}_c${CONCURRENCY}_t${MAX_TOKENS}.json" \
  --output-raw-json "${OUT_DIR}/raw_r${REQUESTS}_c${CONCURRENCY}_t${MAX_TOKENS}.json"
