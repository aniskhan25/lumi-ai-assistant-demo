#!/bin/bash
# Per-user serving sweeps for the "largest models on LUMI" answer, comparable with the
# Kimi-K2.7-Code runs in repro/kimi_k27_aitta/. One model per allocation.
#
#   bash repro/aitta_model_survey/submit.sh                  # all
#   DEPENDENCY=12345 bash repro/aitta_model_survey/submit.sh qwen   # after a download job
#
# Run from the repository root. Results land in benchmarks/results/survey-<name>/job_<id>/.
set -euo pipefail

CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"
WORKLOADS="${WORKLOADS:-8192:512 1024:1024}"
COMMON="--max-model-len 65536 --max-num-seqs 32 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95"
# LUMI's sbatch ignores SBATCH_DEPENDENCY, so pass it explicitly.
DEP_ARG="${DEPENDENCY:+--dependency=afterok:${DEPENDENCY}}"

submit() {
  local name="$1" model="$2" nodes="$3" tp="$4" pp="$5" args="$6"
  MODE=serving BENCH_PROFILE="survey-${name}" MODEL="${model}" \
  TP_SIZE="${tp}" PP_SIZE="${pp}" STARTUP_TIMEOUT_S=5400 \
  CONCURRENCIES="${CONCURRENCIES}" WORKLOADS="${WORKLOADS}" \
  EXTRA_VLLM_ARGS="${COMMON} ${args}" \
    sbatch ${DEP_ARG:+"${DEP_ARG}"} --job-name="survey-${name}" --nodes="${nodes}" --time=04:00:00 \
    run_vllm_bench_multinode.sh
}

# TP stays inside a node and PP crosses nodes: the one hang so far (Kimi A) had TP
# spanning two nodes, while the PP-across-nodes runs were stable.
for x in ${@:-qwen r1 glm kimi}; do
  case "${x}" in
    qwen) submit qwen Qwen/Qwen3-Coder-480B-A35B-Instruct 4 8 4 "" ;;
    r1)   submit r1 deepseek-ai/DeepSeek-R1-0528 2 8 2 "--enable-expert-parallel" ;;
    # GLM-5.x and DeepSeek-V4 need aiter's sparse MLA, which gfx90a lacks
    # (laifs-container-recipes#8); GLM-4.7 is the newest GLM with dense attention.
    glm)  submit glm zai-org/GLM-4.7 2 8 2 "--enable-expert-parallel" ;;
    # Same layout as repro/kimi_k27_aitta/ combination B, now with both workloads.
    kimi) submit kimi moonshotai/Kimi-K2.7-Code 3 8 3 "--trust-remote-code --enable-expert-parallel" ;;
    # Same model as r1 in INT4 (W4A16, 346 GiB): fits one node, so no cross-node collectives,
    # and isolates the FP8 cost on gfx90a, which has no FP8 hardware.
    r1int4) submit r1int4 RedHatAI/DeepSeek-R1-0528-quantized.w4a16 1 8 1 "--enable-expert-parallel" ;;
    *) echo "unknown run: ${x}" >&2; exit 2 ;;
  esac
done
