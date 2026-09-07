#!/bin/bash
# Measure raw read throughput for a checkpoint on /scratch, with no loader and no GPU
# involved. Establishes the floor under any weight-load time: if Lustre delivers
# N GB/s, a checkpoint of S GB cannot possibly load faster than S/N seconds.
#
#   lustre_read.sh <model-id-or-path> <budget_gb>
set -uo pipefail

MODEL="${1:-}"
BUDGET_GB="${2:-64}"
OUT="${RESULTS_DIR:-results}/lustre_read.txt"
mkdir -p "$(dirname "${OUT}")" 2>/dev/null || true

# Resolve an HF id to its snapshot directory in the shared cache.
resolve_dir() {
  local model="$1"
  if [ -d "${model}" ]; then echo "${model}"; return; fi
  local hub="${HF_HOME:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache}/hub"
  local repo="models--${model//\//--}"
  local snap
  snap="$(ls -d "${hub}/${repo}/snapshots/"* 2>/dev/null | head -n 1)"
  [ -n "${snap}" ] && echo "${snap}"
}

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(hostname) ==="
  echo "model: ${MODEL}"

  DIR="$(resolve_dir "${MODEL}")"
  if [ -z "${DIR}" ] || [ ! -d "${DIR}" ]; then
    echo "could not resolve ${MODEL} to a directory; nothing measured"
    exit 0
  fi
  echo "resolved to: ${DIR}"
  echo "filesystem:  $(df -h "${DIR}" 2>/dev/null | tail -1)"
  echo "lustre stripe:"
  lfs getstripe -d "${DIR}" 2>/dev/null | head -5 || echo "  (lfs unavailable)"

  total_bytes="$(du -sbL "${DIR}" 2>/dev/null | cut -f1)"
  echo "checkpoint size: ${total_bytes:-unknown} bytes"

  echo
  echo "--- sequential read, up to ${BUDGET_GB} GB of safetensors shards ---"
  budget=$(( BUDGET_GB * 1024 * 1024 * 1024 ))
  read_bytes=0
  start="$(date +%s.%N)"
  # -L so a symlinked HF snapshot is followed to the real blob.
  while IFS= read -r shard; do
    [ "${read_bytes}" -ge "${budget}" ] && break
    size="$(stat -Lc %s "${shard}" 2>/dev/null || echo 0)"
    [ "${size}" = "0" ] && continue
    # iflag=direct would be ideal but is not always permitted; the page cache is cold
    # for a 959 GB checkpoint anyway, which is the case that matters.
    dd if="${shard}" of=/dev/null bs=16M 2>/dev/null || true
    read_bytes=$(( read_bytes + size ))
  done < <(find -L "${DIR}" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | sort)
  end="$(date +%s.%N)"

  elapsed="$(awk -v a="${start}" -v b="${end}" 'BEGIN{printf "%.2f", b-a}')"
  echo "read ${read_bytes} bytes in ${elapsed}s"
  awk -v n="${read_bytes}" -v t="${elapsed}" 'BEGIN{
    if (t > 0) printf "throughput: %.2f GB/s (%.2f GiB read)\n", n/t/1e9, n/1073741824;
  }'
  if [ -n "${total_bytes}" ] && [ "${read_bytes}" -gt 0 ]; then
    awk -v total="${total_bytes}" -v n="${read_bytes}" -v t="${elapsed}" 'BEGIN{
      if (t > 0 && n > 0)
        printf "implied floor for the whole checkpoint: %.0f s (%.1f min)\n",
               total / (n/t), total / (n/t) / 60;
    }'
  fi
} > "${OUT}" 2>&1

echo "wrote ${OUT}"
cat "${OUT}"
