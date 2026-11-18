"""Split-K sparse attention decode on gfx90a for small batches (DSV4_SPARSE_SPLIT=1).

vLLM 0.26 decodes V4's sparse attention on ROCm with a split-K Triton kernel (partial +
reduce) only on gfx942/gfx950; every other GPU takes a one-pass "fallback for un-tuned
architectures" with a grid of queries x head blocks. With 8 heads per rank at TP=8 that is
one workgroup per query, so at 1 user one workgroup on a 110-CU GCD walks all ~640 KV
entries of a ratio-4 layer: ~0.4 ms per layer, ~10 ms per decode step.

The split path does the same arithmetic split along the KV list and contains nothing
architecture-specific (it decodes the FP8 cache like the fallback). Its per-workgroup loop
is slower than the fallback's, so it only pays while splitting adds parallelism. GPU time
per call on one GCD (test_sparse_split.py, graph replay, job 23638388), fallback -> split:

    queries              1-16          32         64          128
    ratio 4 + top-k   409 -> 61-147  440 -> 274  451 -> 600  898 -> 1256
    ratio 128         100 -> 58-60   102 -> 79   106 -> 212  212 -> 364
    SWA only           88 -> 26-28    92 -> 47    94 -> 78   188 -> 154

In serving (recipe + SPARSE_SPLIT=1, 500 GSM8K unchanged) a cutoff of 32 cut median TPOT
at 1/4/16 users (64.1 -> 60.1, 69.0 -> 65.8, 77.0 -> 74.5 ms) but raised it at 32 users
(96.8 -> 107.1 ms, jobs 23638474/5 vs 23591543/4) despite the kernel-level win above, so
the split path is taken for batches of at most MAX_QUERIES queries (DSV4_SPARSE_SPLIT_MAX,
default 16; same-day A/B runs still lost ~3 ms at 16-32 users with 16). Batch size is fixed
per CUDA graph, so the choice is made at capture. The gate is one module-level flag that the
decode wrapper reads at call time; two indexer paths also read it, so it is set only for the
duration of the decode call.
"""

import os

MAX_QUERIES = int(os.environ.get("DSV4_SPARSE_SPLIT_MAX", "16"))


def install() -> None:
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

    orig = m._rocm_sparse_attn_decode_ragged_triton

    def split_decode(*args, **kwargs):
        q = args[0] if args else kwargs["q"]  # vLLM's caller passes keywords
        if q.shape[0] > MAX_QUERIES:
            return orig(*args, **kwargs)
        saved = m._ON_GFX950
        m._ON_GFX950 = True
        try:
            return orig(*args, **kwargs)
        finally:
            m._ON_GFX950 = saved

    m._rocm_sparse_attn_decode_ragged_triton = split_decode
