"""CPU check of the backported `_shard_fp8_qkv_proj` on MiMo-V2.6 shapes.

Builds a fused qkv_proj in the checkpoint's layout -- pre-sharded into
`ckpt_tp` chunks of [Q | K | V], FP8 with per-chunk 128x128 block scales --
and checks that every rank's slice, dequantised, equals that rank's rows of
the original matrix in the de-interleaved [Q.. | K.. | V..] order the forward
expects. Run inside the container:

    singularity exec <sif> python3 repro/lexivo_mimo_gfx90a/test_shard.py <mimo_v2.py>
"""

import importlib.util
import sys

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_quantize,
)
from vllm.utils.math_utils import cdiv

spec = importlib.util.spec_from_file_location(
    "vllm.model_executor.models.mimo_v2_patched", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BLOCK, HIDDEN, NH, HD, VD, CKPT_TP = 128, 512, 64, 192, 128, 4
FP8 = torch.float8_e4m3fn


def quantize_chunk(w):
    rows = w.shape[0]
    pad = cdiv(rows, BLOCK) * BLOCK - rows
    wq, s = scaled_quantize(torch.cat([w, w.new_zeros(pad, w.shape[1])]),
                            GroupShape(BLOCK, BLOCK), FP8, compute_dtype=torch.float32)
    return wq[:rows], s


def dequant(wq, s):
    return wq.float() * s.repeat_interleave(BLOCK, 0)[: wq.shape[0]].repeat_interleave(BLOCK, 1)


def check(nkv, tp):
    torch.manual_seed(0)
    q, k, v = NH // CKPT_TP * HD, nkv // CKPT_TP * HD, nkv // CKPT_TP * VD
    chunks = [torch.randn(q + k + v, HIDDEN) for _ in range(CKPT_TP)]
    quant = [quantize_chunk(c) for c in chunks]
    w_full = torch.cat([wq for wq, _ in quant])
    s_full = torch.cat([s for _, s in quant])
    # The exact values the loader sees, as one float matrix per chunk.
    ref = [dequant(wq, s) for wq, s in quant]

    def head_rows(kind, h):  # rows of head h of Q/K/V in the reference
        per = {"q": NH // CKPT_TP, "k": nkv // CKPT_TP, "v": nkv // CKPT_TP}[kind]
        c, i = divmod(h, per)
        off, dim = {"q": (0, HD), "k": (q, HD), "v": (q + k, VD)}[kind]
        return ref[c][off + i * dim: off + (i + 1) * dim]

    worst = 0.0
    for r in range(tp):
        wq, s = mod._shard_fp8_qkv_proj(w_full, s_full, NH, nkv, HD, VD, r, tp, CKPT_TP)
        got = dequant(wq, s)
        kv_ids = (list(range(r * nkv // tp, (r + 1) * nkv // tp)) if tp <= nkv
                  else [r // (tp // nkv)])
        want = torch.cat(
            [head_rows("q", h) for h in range(r * NH // tp, (r + 1) * NH // tp)]
            + [head_rows("k", h) for h in kv_ids] + [head_rows("v", h) for h in kv_ids])
        assert got.shape == want.shape, (got.shape, want.shape)
        worst = max(worst, ((got - want).abs().max() / want.abs().max()).item())
    # A plain chunk is exact; a re-quantised slice carries one FP8 rounding (~6%).
    ok = worst < 0.07
    print(f"kv_heads={nkv} tp={tp}: max rel err {worst:.4f} {'OK' if ok else 'FAIL'}")
    return ok


results = [check(nkv, tp) for nkv in (4, 8) for tp in (2, 4, 8)]
sys.exit(0 if all(results) else 1)
