"""Sparse-attention prefill launch options (DSV4_SPARSE_PREFILL=1), from the V4.1 branch.

vLLM 0.26's `_rocm_sparse_attn_prefill_ragged_triton` (rocm_aiter_mla_sparse.py) launches
`_sparse_attn_prefill_ragged_kernel` with 4 warps and Triton's default num_stages (2). On
gfx90a, software-pipelining the gathered KV rows makes it 2.2x slower, and 2 warps with AMD's
16-wide matrix instruction help again: the V4.1 port measured 5.99 -> 2.87 -> 1.77 ms per
2048-query chunk at 16k context, same tile and output (lumi_dsv41/hooks.py
_prefill_stages, jobs 23900392, 23983859). It is ~23 % of a V4 prefill (profile job
24005447: 2.8 ms a call, 44 calls per 2048-token chunk). Same text patch as there.
"""


def _source_patch(module, name, old, new) -> None:
    """Re-define module.name from its own source with one exact text replacement."""
    import inspect
    import textwrap

    target = getattr(module, name)
    source = textwrap.dedent(inspect.getsource(target))
    if source.count(old) != 1:
        raise RuntimeError(f"lumi_dsv4_debug: cannot patch {module.__name__}.{name}: "
                           f"{old!r} found {source.count(old)} times")
    scope = {}
    exec(compile(source.replace(old, new), inspect.getsourcefile(module), "exec"),
         module.__dict__, scope)
    setattr(module, name, scope[name])


def install() -> None:
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

    if getattr(m, "_ON_GFX950", False):
        return
    _source_patch(
        m, "_rocm_sparse_attn_prefill_ragged_triton",
        "        num_warps=num_warps,\n",
        "        num_warps=2,\n        num_stages=1,\n        matrix_instr_nonkdim=16,\n"
        "        waves_per_eu=1,\n",
    )
