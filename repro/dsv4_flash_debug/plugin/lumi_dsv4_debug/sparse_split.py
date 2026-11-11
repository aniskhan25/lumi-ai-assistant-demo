"""Split-K sparse attention decode on gfx90a (DSV4_SPARSE_SPLIT=1).

vLLM 0.26 decodes V4's sparse attention on ROCm with a split-K Triton kernel (partial +
reduce) only on gfx942/gfx950; every other GPU takes a one-pass "fallback for un-tuned
architectures" with a grid of queries x head blocks. With 8 heads per rank at TP=8 that is
one workgroup per query, so at 1 user one workgroup on a 110-CU GCD walks all ~640 KV
entries: ~0.24 ms per layer, 10 ms per decode step (eager profile, job 23598813).

The split path does the same arithmetic split along the KV list and contains nothing
architecture-specific (it decodes the FP8 cache like the fallback). The gate is one
module-level flag that the decode wrapper reads at call time; it is also read by two
indexer paths, so it is set only for the duration of the decode call.
"""


def install() -> None:
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

    orig = m._rocm_sparse_attn_decode_ragged_triton

    def split_decode(*args, **kwargs):
        saved = m._ON_GFX950
        m._ON_GFX950 = True
        try:
            return orig(*args, **kwargs)
        finally:
            m._ON_GFX950 = saved

    m._rocm_sparse_attn_decode_ragged_triton = split_decode
