#!/bin/bash
# Capture everything a LUMI/RCCL bug report needs. Runs inside the container.
# Never fails the job: every probe is best effort.
set -uo pipefail

OUT="${RESULTS_DIR:-$(dirname "$0")/results}/env.txt"
mkdir -p "$(dirname "${OUT}")" 2>/dev/null || true

{
  echo "=== when / where ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "host: $(hostname)"
  echo "slurm job: ${SLURM_JOB_ID:-none}  account: ${SLURM_JOB_ACCOUNT:-none}"
  echo "nodes: ${SLURM_JOB_NUM_NODES:-?}  tasks/node: ${SLURM_NTASKS_PER_NODE:-?}  world: ${SLURM_NPROCS:-?}"
  echo "container: ${CONTAINER:-unset}"
  # A floating `latest` symlink would make these timings unreproducible, so record what
  # the pinned path actually resolved to.
  if [ -n "${CONTAINER:-}" ]; then
    echo "container resolves to: $(readlink -f "${CONTAINER}" 2>/dev/null || echo "unresolved")"
  fi
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"

  echo
  echo "=== interconnect ==="
  # The heart of hypothesis 1: several interfaces exist, and RCCL picks one itself
  # unless NCCL_SOCKET_IFNAME says otherwise.
  echo "--- interfaces ---"
  ip -o link show 2>/dev/null | awk -F': ' '{print $2}' || echo "ip unavailable"
  echo "--- cxi devices ---"
  ls /sys/class/cxi/ 2>/dev/null || echo "no /sys/class/cxi"
  echo "--- libfabric cxi provider ---"
  for fi in /opt/cray/libfabric/*/bin/fi_info "$(command -v fi_info 2>/dev/null)"; do
    [ -x "${fi}" ] || continue
    echo "using ${fi}"
    "${fi}" -p cxi 2>&1 | head -40
    break
  done
  echo "--- rccl net plugin ---"
  ls -l /usr/lib/x86_64-linux-gnu/librccl-net* 2>/dev/null || echo "no librccl-net in the usual place"
  ls -l /opt/rocm/lib/librccl.so* 2>/dev/null || echo "no librccl"

  echo
  echo "=== gpu / rocm ==="
  rocminfo 2>/dev/null | grep -m 8 -i "gfx" || echo "rocminfo unavailable"
  hipconfig --version 2>/dev/null || echo "hipconfig unavailable"
  rocm-smi --showproductname --showdriverversion 2>/dev/null || echo "rocm-smi unavailable"

  echo
  echo "=== python packages ==="
  python3 - <<'PY'
for name in ("torch", "vllm", "transformers", "triton"):
    try:
        module = __import__(name)
        print(f"{name:16s} {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name:16s} <not importable: {type(exc).__name__}>")
try:
    import torch
    print(f"torch.hip       {torch.version.hip}")
    print(f"nccl/rccl       {torch.cuda.nccl.version()}")
    print(f"gpu             {torch.cuda.get_device_name(0)}")
    print(f"gcn arch        {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"device count    {torch.cuda.device_count()}")
except Exception as exc:
    print(f"torch device probe failed: {exc!r}")
PY

  echo
  echo "=== comms / runtime environment variables ==="
  # The repo's other collector omits NCCL_/RCCL_/FI_, which is why it could never say
  # which transport RCCL chose. Do not narrow this filter.
  env | grep -E '^(NCCL_|RCCL_|FI_|OFI_|CXI_|VLLM_|RUNAI_STREAMER_|HSA_|HIP_|ROCR_|MIOPEN_|TORCH|SLURM_MPI)' |
    sort || echo "none set"
} > "${OUT}" 2>&1

echo "wrote ${OUT}"
cat "${OUT}"
