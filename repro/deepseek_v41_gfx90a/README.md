# DeepSeek-V4.1-Flash on LUMI-G (MI250X, gfx90a): phase 0

Should we try to port `deepseek-ai/DeepSeek-V4.1-Flash` to LAIF's vLLM 0.26, as we did for
Qwen3.8-Flash (branch `exp/qwen38-flash-gfx90a`)? This is the de-risking pass, before any
port work and before downloading its 475 GiB.

## The model (config of 2026-09-10)

`DeepseekV41ForCausalLM`, 40 layers, hidden 5120, 384 routed experts with 6 active plus 1
shared.

- **Attention:** MLA with one KV head of width 512, of which 64 are rope dimensions. It is
  compressed sparse attention: per-layer compress ratios 2/1/0, a 128-token sliding window,
  and an indexer with 32 heads that keeps the top 512. On top of DeepSeek-V4 it adds
  cross-layer K/V sharing (`kv_source_layer_ids`) and candidate-block preselection.
- **Other:** 4-way hyper-connections with Sinkhorn normalisation, and Engram n-gram tables at
  layers 1 and 14 (the same idea as Qwen3.8's PLE, and upstream shares code between them).
  There are also DSpark/MTP draft layers.
- **Number formats:** FP8 weights in 32×32 blocks with `ue8m0` scales, and FP4 (MXFP4)
  experts. gfx90a has no FP8 or FP4 units, so both would be dequantised in software.
- **Size: 475 GiB. That does not fit in one LUMI-G node** (8 × 64 GB HBM), so it needs at
  least two nodes (TP=16 or PP=2), and the plugin approach has not ported PP.

## Static findings (vLLM main @ 790be38 vs LAIF's vLLM 0.26.0)

**Not in 0.26.** `DeepseekV41ForCausalLM` is absent; 0.26 has DeepSeek-V4 and V3.2. Upstream
has `vllm/models/deepseek_v41/{amd,nvidia,common}`, laid out like Qwen3.8's package.

**aiter on the default AMD path.** The ROCm attention (`amd/rocm.py`, class
`DeepseekV41ROCMAiterMLAAttention`) branches on `_ON_GFX950` and `rocm_aiter_ops.is_enabled()`,
and it has explicit non-gfx950, non-aiter branches. The ops it uses have Triton or torch
fallbacks:
- sparse prefill: Triton after the aiter variants;
- sparse decode: Triton, with a "fallback decode kernel" for architectures other than
  gfx942/gfx950;
- indexer logits: Triton and torch.

The MXFP4 indexer (`rocm_paged_mxfp4_indexer`) is "gfx950 only" and aiter-based, so gfx90a
must take the FP8/BF16 indexer. The optional monokernel decode (`amd/mono`, 9k lines) imports
aiter FlyDSL directly, but it is opt-in (`--monokernel`, gfx942/gfx950). So aiter is not a
hard wall, but nothing upstream tests the non-gfx94x/95x branches.

**The port would be much larger than Qwen3.8's:**

| | Qwen3.8-Flash | DeepSeek-V4.1-Flash |
|---|---|---|
| Upstream model code to carry (no NVIDIA, no monokernel) | ~9.4k lines | ~15k lines |
| Modules missing from 0.26 | 4 | **17** |
| Names missing from existing 0.26 modules | ~15 | **~48** |
| Core vLLM files whose newer version is needed | none | **the sparse-MLA stack, shared with 0.26's own V4** |

The core files have roughly doubled since 0.26, and V4.1 imports new names from them, so
their newer versions would have to replace 0.26's:

| file | 0.26 | main | lines changed |
|---|---|---|---|
| `v1/attention/ops/rocm_aiter_mla_sparse.py` | 2,400 | 4,986 | 3,018 |
| `v1/attention/backends/mla/indexer.py` | 960 | 2,053 | 1,721 |
| `model_executor/layers/sparse_attn_indexer.py` | 849 | 1,692 | 1,117 |
| `v1/attention/backends/mla/sparse_swa.py` | 782 | 1,393 | 945 |
| `models/deepseek_v4/attention.py` | 807 | 1,185 | 672 |
| `model_executor/layers/mhc.py` | 568 | 1,020 | 498 |
| `v1/attention/backend.py` | 1,110 | 1,160 | 486 |
| `compressor_utils.py`, plus 3 files new since 0.26 | | | ~1,700 |

That is roughly 10k changed lines in core attention code, against Qwen3.8's ~735 lines of
bridging and no core replacements.

## Runtime checks

See below. They test the parts V4.1 shares with DeepSeek-V4, which 0.26 supports natively:

1. 0.26's own ROCm kernel tests for the DSV4 sparse-MLA ops on one GCD
   (`run_kernel_tests.sh`, tests under `vendor/tests_v026`, unmodified).
2. DeepSeek-V4-Flash (149 GiB, fits one node) served by the stock container with the
   correctness probe (`run_v4_serve.sh`).
