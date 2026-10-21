#!/bin/bash
# MiMo-V2.6-Flash on one LUMI-G node: does it produce correct output, and in which
# container? One variant per allocation.
#
#   bash repro/lexivo_mimo_gfx90a/submit.sh new-tu    # LAIF 20260929, vLLM 0.26.0, recipe
#   bash repro/lexivo_mimo_gfx90a/submit.sh old       # LAIF 20260807, vLLM 0.22.1, as reported
#
# Run from the repository root. Results land in benchmarks/results/mimo-<name>/job_<id>/.
set -euo pipefail

MODEL="${MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-MOPD}"
NEW=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif
OLD=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
GSM8K="$(ls /scratch/project_462000131/${USER}/hf-cache/hub/datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet | head -n1)"
DEP_ARG="${DEPENDENCY:+--dependency=afterok:${DEPENDENCY}}"

submit() {
  local name="$1" container="$2" mode="$3" time="$4" args="$5"
  MODE="${mode}" BENCH_PROFILE="mimo-${name}" MODEL="${MODEL}" CONTAINER="${container}" \
  TP_SIZE=8 PP_SIZE=1 STARTUP_TIMEOUT_S=3600 PROBE_GSM8K_PARQUET="${GSM8K}" \
  CONCURRENCIES="1 4 16 32" WORKLOADS="1024:1024 8192:512" \
  EXTRA_VLLM_ARGS="${args}" \
    sbatch ${DEP_ARG:+"${DEP_ARG}"} --job-name="mimo-${name}" --nodes=1 --time="${time}" \
    run_vllm_bench_multinode.sh
}

# No --attention-backend: every MiMo layer has head_dim 192 / v_head_dim 128, and
# vLLM 0.26 routes those to TRITON_ATTN_DIFFKV, which supports the SWA sinks.
NEW_ARGS="--trust-remote-code --max-model-len 65536 --max-num-seqs 32 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95"

for x in ${@:-new-tu old}; do
  case "${x}" in
    # The experts are stored as MXFP4 (store_dtype). On ROCm vLLM's auto choice for
    # them is aiter only, so without --moe-backend startup fails (job 23017831).
    new-tu)     submit new-tu "${NEW}" serving 03:00:00 "${NEW_ARGS} --moe-backend triton_unfused" ;;
    new-triton) submit new-triton "${NEW}" serving 03:00:00 "${NEW_ARGS} --moe-backend triton" ;;
    # Dequantises the experts to bf16: slow, but the correctness reference.
    new-emu)    submit new-emu "${NEW}" smoke 02:00:00 "${NEW_ARGS} --moe-backend emulation" ;;
    # Lexivo's reported setup: vLLM 0.22.1, bf16, ROCM_ATTN, TP=8 on one node.
    old) submit old "${OLD}" smoke 01:30:00 \
           "--trust-remote-code --dtype bfloat16 --attention-backend ROCM_ATTN --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.95" ;;
    *) echo "unknown run: ${x}" >&2; exit 2 ;;
  esac
done
