#!/bin/bash
# Kimi-K2.7-Code (native INT4 experts, ~577 GB) for interactive Aitta serving on LUMI-G.
# Four combinations, one per allocation -- back-to-back variants in one allocation are not
# comparable, because the first pays the one-time costs.
#
#   bash repro/kimi_k27_aitta/submit.sh          # all four
#   bash repro/kimi_k27_aitta/submit.sh A C      # a subset
#
# Run from the repository root. Results land in benchmarks/results/kimi27-<X>/job_<id>/.
set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-K2.7-Code}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"
COMMON="--trust-remote-code --gpu-memory-utilization 0.95"

submit() {
  local name="$1" nodes="$2" tp="$3" pp="$4" walltime="$5" startup="$6" args="$7"
  MODE=serving BENCH_PROFILE="kimi27-${name}" MODEL="${MODEL}" \
  TP_SIZE="${tp}" PP_SIZE="${pp}" STARTUP_TIMEOUT_S="${startup}" \
  CONCURRENCIES="${CONCURRENCIES}" EXTRA_VLLM_ARGS="${COMMON} ${args}" \
    sbatch --job-name="kimi27-${name}" --nodes="${nodes}" --time="${walltime}" \
    run_vllm_bench_multinode.sh
}

for combo in "${@:-A B C D}"; do
  for x in ${combo}; do
    case "${x}" in
      # Cheapest: does it fit on 2 nodes, with no pipeline bubble?
      A) submit A 2 16 1 03:00:00 5400 \
           "--enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 4096 --max-model-len 65536" ;;
      # Safe fit: the fallback if A does not fit, and what PP=3 costs per user.
      B) submit B 3 8 3 03:00:00 5400 \
           "--enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 4096 --max-model-len 65536" ;;
      # Fastest per user: fewer slots, smaller prefill chunks, no EP. Without EP every rank
      # loads its TP shard of all experts, which took ~2 h for the FP8 Kimi -- hence 4 h.
      C) submit C 2 16 1 04:00:00 9000 \
           "--max-num-seqs 16 --max-num-batched-tokens 2048 --max-model-len 65536" ;;
      # Long context: whole-repository prompts, at the cost of KV room.
      D) submit D 3 8 3 03:00:00 5400 \
           "--enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 4096 --max-model-len 131072" ;;
      *) echo "unknown combination: ${x}" >&2; exit 2 ;;
    esac
  done
done
