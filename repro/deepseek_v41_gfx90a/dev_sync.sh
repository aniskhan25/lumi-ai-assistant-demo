#!/bin/bash
# Ship the branch to the LUMI worktree (GitHub is not reachable from LUMI): commit first.
set -euo pipefail
B="${TMPDIR:-/tmp}/dsv41.bundle"
git bundle create -q "$B" HEAD~20..exp/deepseek-v41-gfx90a 2>/dev/null || git bundle create -q "$B" exp/deepseek-v41-gfx90a
scp -q "$B" lumi.csc.fi:/scratch/project_462000131/anisrahm/dsv41.bundle
ssh -o BatchMode=yes lumi.csc.fi 'cd /scratch/project_462000131/anisrahm/lumi-ai-assistant-demo-dsv41 && git fetch -q ../dsv41.bundle exp/deepseek-v41-gfx90a:refs/bundle/dsv41 && git merge -q --ff-only refs/bundle/dsv41 && rm ../dsv41.bundle && git log --oneline -1'
