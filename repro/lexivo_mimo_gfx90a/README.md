# MiMo-V2.6-Flash on LUMI-G (MI250X, gfx90a)

Prompted by a user report (Lexivo, 6 Oct 2026): MiMo-V2.6-Flash generated garbage
(",,,,") on LUMI-G with LAIF 20260807 (vLLM 0.22.1), `ROCM_ATTN`, TP=8. Checkpoint:
`XiaomiMiMo/MiMo-V2.6-Flash-MOPD` (166 GiB; experts MXFP4 inside an FP8 config,
`store_dtype: mxfp4`; head_dim 192 / v_head_dim 128; attention sinks on the 39
sliding-window layers; 4 KV heads on the 9 global layers, 8 on the SWA layers).

**Result: it runs correctly on one node** with LAIF 20260929 (vLLM 0.26.0), four flags,
and one backported vLLM source file. GSM8K 97.5 % (200 questions, greedy, thinking off),
long-context retrieval at 16.8k tokens, no NaN or garbage. ~72 ms/token for one user.

## Recipe

```bash
#SBATCH --partition=standard-g --nodes=1 --gpus-per-node=8 --mem=460G
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif
VLLM_PY=/opt/venv/lib/python3.12/site-packages/vllm
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"

singularity exec --bind "$PWD/mimo_v2.py:${VLLM_PY}/model_executor/models/mimo_v2.py" "$SIF" \
  vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --language-model-only \
    --moe-backend triton_unfused \
    --max-model-len 65536 --max-num-seqs 32 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95
```

`mimo_v2.py` is the container's file with one function backported from vLLM main
(`backport_mimo_v2.py` regenerates it; `test_shard.py` checks it on CPU). In this repo:
`bash repro/lexivo_mimo_gfx90a/submit.sh new-tu-tp8`.

Untested: thinking-mode output parsing (`--reasoning-parser mimo`, `--tool-call-parser
mimo` from Xiaomi's recipe), MTP/speculative decoding, image/audio input.

## What each piece fixes (one allocation per attempt)

| job | setup | outcome |
|---|---|---|
| 23017832 | LAIF 20260807, vLLM 0.22.1, Lexivo's flags | `AssertionError: Sinks are only supported in FlashAttention 3`. 0.22.1 hard-wires `FlashAttentionDiffKVBackend` for 192/128 heads; ROCm's flash-attn 2.8.3 has no sinks. `--attention-backend ROCM_ATTN` does not reach these layers. Stock 0.22.1 cannot start this model, so the reported garbage came from a local workaround. |
| 23017831 | 0.26.0, no `--moe-backend` | Attention is fine (`TRITON_ATTN_DIFFKV`, supports sinks). `No MXFP4 MoE backend supports the deployment configuration`: on ROCm the auto list is aiter only. |
| 23017920 | `--moe-backend triton` | The gpt-oss kernel: `does not support MoEActivation.SILU`. |
| 23017921 | `--moe-backend emulation` | Not implemented for `Mxfp4MoEMethod` (seen once loading got further, job 23018152). |
| 23017919 | `triton_unfused`, TP=8 | `TP size must evenly split the number of KV heads` (global layers have 4). |
| 23017976, 23017979 | `triton_unfused`, TP=4 (DP=2 / PP=2) | `size of tensor a (1856) must match ... (1792)`. vLLM 0.26 assumes the fused FP8 qkv_proj is one [Q\|K\|V] group per KV head; the checkpoint is pre-sharded into 4 chunks with per-chunk block scales. Every TP size fails (`test_shard.py` against the stock file reproduces both errors). Fixed upstream; backported. |
| 23018151 | patched, TP=8 | Weights load; the profiling run dies in the vision encoder, which imports CUDA-only `vllm_flash_attn`. → `--language-model-only`. |
| **23018587** | **patched, TP=8, LM only** | **Works.** Startup 321 s (weights 88 s). |
| 23018586 | patched, TP=4 DP=2, LM only | Works. Startup 407 s. Slower than TP=8 at every point. |

## Results (2026-10-21)

Correctness probe (`probe.py`, greedy, `enable_thinking: false`):

| job | sanity (5) | needle 2.1k / 16.8k tok | GSM8K (200) | non-finite logprobs | garbage |
|---|---|---|---|---|---|
| TP=8 (23018587) | 5/5 | ok / ok | 97.5 % | 0 | 0 |
| TP=4 DP=2 (23018586) | 4/5 | ok / ok | 97.5 % | 0 | 0 |

The DP=2 sanity miss is factual, not corruption: asked for the Finnish employment
contracts act it answered the repealed 1946 act instead of Työsopimuslaki (2001/55).
The needle sits 16.8k tokens before the question, far outside the 128-token window,
so only the global layers can carry it.

Streaming `vllm bench serve`; TPOT median/p99 in ms, TTFT p99, total output tok/s.
Every request completed.

| layout | workload | 1 user | 4 users | 16 users | 32 users |
|---|---|---|---|---|---|
| TP=8 | 1024:1024 | **72**/72, 0.09 s, 14 | 74/74, 0.9 s, 54 | 82/83, 3.1 s, 193 | 96/102, 5.7 s, 323 |
| | 8192:512 | 72/72, 1.4 s, 13 | 74/81, 5.3 s, 50 | 113/117, 18.7 s, 145 | 103/168, 36.5 s, 235 |
| TP=4 DP=2 | 1024:1024 | 77/77, 0.53 s, 13 | 77/77, 1.4 s, 52 | 88/91, 3.6 s, 178 | 100/104, 5.3 s, 311 |
| | 8192:512 | 78/78, 2.5 s, 12 | 78/88, 8.3 s, 46 | 118/129, 19.0 s, 127 | 130/190, 35.8 s, 194 |

KV room for 65,536-token requests: 66x at TP=8 (4.3 M tokens), 45x per DP replica.
Per user this is ~1.8x slower than GLM-4.7 (40 ms, `repro/aitta_model_survey/`) on
half the hardware (1 node vs 2): the MXFP4 experts run through the unfused Triton
path, dequantised on the fly, since gfx90a has no FP4/FP8 units.
