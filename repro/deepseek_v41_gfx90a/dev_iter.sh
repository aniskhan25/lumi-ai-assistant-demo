#!/bin/bash
# Local: commit, ship to LUMI, run a dev.sh command there and show the tail.
#   repro/deepseek_v41_gfx90a/dev_iter.sh "msg" python repro/deepseek_v41_gfx90a/check_import.py
set -euo pipefail
MSG="$1"; shift
git add -A repro/deepseek_v41_gfx90a && { git diff --cached --quiet || git commit -qm "${MSG}"; }
"$(dirname "$0")/dev_sync.sh"
ssh -o BatchMode=yes lumi.csc.fi "cd /scratch/project_462000131/anisrahm/lumi-ai-assistant-demo-dsv41 && timeout ${TIMEOUT:-900} repro/deepseek_v41_gfx90a/dev.sh $* 2>&1 | grep -vE 'amdsmi|pynvml|FutureWarning|^\s+import pynvml' | tail -${TAIL:-25}"
