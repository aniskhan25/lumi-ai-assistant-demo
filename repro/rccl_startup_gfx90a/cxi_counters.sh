#!/bin/bash
# Snapshot this node's Cassini (Slingshot) telemetry. Never fails the job: every probe
# is best effort, and a missing counter is itself worth recording.
#
#   cxi_counters.sh <variant> <before|after>
set -uo pipefail

VARIANT="${1:-baseline}"
WHEN="${2:-before}"
OUT="${RESULTS_DIR:-results}/cxi_${VARIANT}_$(hostname)_${WHEN}.txt"
mkdir -p "$(dirname "${OUT}")" 2>/dev/null || true

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)  host=$(hostname)  variant=${VARIANT}  when=${WHEN} ==="

  echo
  echo "--- interfaces (which ones RCCL could pick) ---"
  ip -o link show 2>/dev/null | awk -F': ' '{print $2}' || echo "ip unavailable"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}"

  echo
  echo "--- cxi_stat ---"
  cxi_stat 2>&1 || echo "cxi_stat unavailable"

  echo
  echo "--- raw telemetry counters ---"
  # The discriminating counters for this case: completion/event queue overflow, portal
  # table entry exhaustion, and retry storms. A stall from connection setup shows up in
  # the retry/setup counters while CQ-full stays flat -- which is what the reporter's
  # null result on FI_CXI_DEFAULT_CQ_SIZE already predicts.
  for dev in /sys/class/cxi/cxi*; do
    [ -d "${dev}" ] || continue
    echo "[$(basename "${dev}")]"
    find "${dev}" -maxdepth 3 -type f \
      \( -name "*cq*" -o -name "*eq*" -o -name "*pt_te*" -o -name "*retry*" \
         -o -name "*no_matching*" -o -name "*drop*" -o -name "*full*" \) 2>/dev/null |
      while read -r counter; do
        value="$(cat "${counter}" 2>/dev/null | tr -d '\n')"
        [ -n "${value}" ] && echo "  ${counter#"${dev}"/} = ${value}"
      done
  done
} > "${OUT}" 2>&1

echo "wrote ${OUT}"
