"""Correct mHC "big fuse" pre kernels on gfx90a, keeping vLLM's fused TileLang mHC path.

Only the layer_input stage of vLLM's TileLang mHC pre kernels is wrong on gfx90a
(test_mhc_tilelang.py, job 23509871); the prenorm GEMM, post, fused post+GEMM and hc_head
kernels are right. The three `mhc_pre_big_fuse*_tilelang` kernels are written for 32-wide
warps: 96 threads per token, threads 0-31 run Sinkhorn on the mixing weights while threads
32-95 apply the pre weights to the residual streams. On a 64-wide wavefront that split cuts
wave 0 in half.

DSV4_MHC_FIX selects a replacement (vLLM's wrappers import these kernels at call time, so
patching the module attribute is enough):

  wave64  the same TileLang kernels with 128 threads split at 64 (one wavefront per role)
  triton  mhc_pre_big_fuse_triton below; only the norm-less kernel, which is the one
          DeepSeek-V4's AMD decoder layer uses
"""

import os
import pathlib

import torch

from vllm.triton_utils import tl, triton

BLOCK_H = 512


@triton.jit
def _pre_big_fuse_kernel(
    mul_ptr, sqr_ptr, scale_ptr, base_ptr, res_ptr, post_ptr, comb_ptr, out_ptr,
    num_tokens, stride_rt, stride_rm,
    rms_eps, pre_eps, sk_eps, post_mult,
    H: tl.constexpr, HC: tl.constexpr, N_SPLITS: tl.constexpr, SK_REPEAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (token, hidden block). Every program recomputes the 4+4+16 mixes
    (cheap); block 0 also writes post_mix and the Sinkhorn-normalised comb_mix."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    HC3: tl.constexpr = HC * (2 + HC)
    j = tl.arange(0, HC)
    jk = j[:, None] * HC + j[None, :]

    sq = tl.load(sqr_ptr + t)
    m_pre = tl.load(mul_ptr + t * HC3 + j)
    for s in tl.static_range(1, N_SPLITS):
        sq += tl.load(sqr_ptr + s * num_tokens + t)
        m_pre += tl.load(mul_ptr + (s * num_tokens + t) * HC3 + j)
    rms = tl.rsqrt(sq / (HC * H) + rms_eps)
    pre = tl.sigmoid(m_pre * rms * tl.load(scale_ptr) + tl.load(base_ptr + j)) + pre_eps

    if hb == 0:
        m_post = tl.load(mul_ptr + t * HC3 + HC + j)
        m_comb = tl.load(mul_ptr + t * HC3 + 2 * HC + jk)
        for s in tl.static_range(1, N_SPLITS):
            m_post += tl.load(mul_ptr + (s * num_tokens + t) * HC3 + HC + j)
            m_comb += tl.load(mul_ptr + (s * num_tokens + t) * HC3 + 2 * HC + jk)
        post = tl.sigmoid(m_post * rms * tl.load(scale_ptr + 1) + tl.load(base_ptr + HC + j))
        tl.store(post_ptr + t * HC + j, post * post_mult)

        cm = m_comb * rms * tl.load(scale_ptr + 2) + tl.load(base_ptr + 2 * HC + jk)
        cm = tl.exp(cm - tl.max(cm, 1)[:, None])
        cm = cm / tl.sum(cm, 1)[:, None] + sk_eps
        cm = cm / (tl.sum(cm, 0)[None, :] + sk_eps)
        for _ in tl.static_range(SK_REPEAT - 1):
            cm = cm / (tl.sum(cm, 1)[:, None] + sk_eps)
            cm = cm / (tl.sum(cm, 0)[None, :] + sk_eps)
        tl.store(comb_ptr + t * HC * HC + jk, cm)

    offs = hb * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(res_ptr + t * stride_rt + j[:, None] * stride_rm + offs[None, :]).to(tl.float32)
    tl.store(out_ptr + t * H + offs, tl.sum(pre[:, None] * x, 0).to(tl.bfloat16))


def mhc_pre_big_fuse_triton(
    gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, post_mix, comb_mix,
    layer_input, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
    sinkhorn_repeat, n_splits=16, hc_mult=4,
):
    """Drop-in for vllm's mhc_pre_big_fuse_tilelang (same positional arguments)."""
    num_tokens = residual.shape[0]
    if num_tokens == 0:
        return
    assert hidden_size % BLOCK_H == 0 and residual.stride(-1) == 1
    assert gemm_out_mul.is_contiguous() and gemm_out_sqrsum.is_contiguous()
    _pre_big_fuse_kernel[(num_tokens, hidden_size // BLOCK_H)](
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, post_mix, comb_mix,
        layer_input, num_tokens, residual.stride(0), residual.stride(1),
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        H=hidden_size, HC=hc_mult, N_SPLITS=gemm_out_mul.shape[0], SK_REPEAT=sinkhorn_repeat,
        BLOCK=BLOCK_H, num_warps=4,
    )


def wave64_tilelang_module():
    """vLLM's tilelang_kernels.py with the three big-fuse kernels at 128 threads split at 64.
    Loaded from a real file: TileLang parses kernel source with inspect."""
    import importlib.util

    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk

    src = pathlib.Path(tk.__file__).read_text()
    old_thr, old_split = "T.Kernel(num_tokens, threads=96)", "T.get_thread_binding() < 32"
    assert src.count(old_thr) == 3 and src.count(old_split) == 3, "unexpected tilelang_kernels.py"
    src = src.replace(old_thr, "T.Kernel(num_tokens, threads=128)").replace(
        old_split, "T.get_thread_binding() < 64")
    root = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"lumi_mhc_wave64_{os.getuid()}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"tilelang_kernels_wave64_{os.getpid()}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"lumi_tilelang_kernels_wave64_{os.getpid()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install(mode: str) -> None:
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk

    if mode == "triton":
        tk.mhc_pre_big_fuse_tilelang = mhc_pre_big_fuse_triton
    elif mode == "wave64":
        mod = wave64_tilelang_module()
        for name in ("mhc_pre_big_fuse_tilelang", "mhc_pre_big_fuse_with_norm_tilelang",
                     "mhc_pre_big_fuse_broadcast_with_norm_tilelang"):
            setattr(tk, name, getattr(mod, name))
    else:
        raise ValueError(f"DSV4_MHC_FIX={mode!r}: expected triton or wave64")


def install_check() -> None:
    """DSV4_MHC_CHECK=1 (eager only: it syncs): recompute every mHC op call with the reference
    that served V4 correctly (torch for pre/post/fused, Triton for hc_head) and log calls whose
    inputs, outputs or reference hold NaN/inf, or whose outputs differ from the reference.
    The served outputs are returned unchanged."""
    import vllm.model_executor.layers.mhc as m
    from vllm.logger import init_logger

    log = init_logger("vllm.lumi_dsv4_debug")
    limit = int(os.environ.get("DSV4_MHC_CHECK_LIMIT", "40"))
    state = {"calls": 0, "events": 0}

    def rows(t: torch.Tensor, n: int) -> torch.Tensor:
        return t.detach().float().reshape(n, -1)

    def check(op, inputs, outs, refs):
        state["calls"] += 1
        if state["events"] >= limit:
            return
        # first input: residual (T, hc, H) for pre/head, layer output x (T, H) for post/fused
        lead = inputs[0]
        n = lead.numel() // (lead.shape[-1] * (lead.shape[-2] if op in ("pre", "head") else 1))
        if n == 0:
            return
        bad_in = [i for i, t in enumerate(inputs)
                  if isinstance(t, torch.Tensor) and t.is_floating_point() and not torch.isfinite(t).all()]
        msgs = []
        for k, (o, r) in enumerate(zip(outs, refs)):
            o2, r2 = rows(o, n), rows(r, n)
            o_bad = (~torch.isfinite(o2)).any(-1)
            r_bad = (~torch.isfinite(r2)).any(-1)
            ok = ~(o_bad | r_bad)
            rel = 0.0
            if ok.any():
                d = (o2[ok] - r2[ok]).norm() / r2[ok].norm().clamp_min(1e-12)
                rel = d.item()
            if o_bad.any() or r_bad.any() or rel > 2e-2:
                first = int(torch.nonzero(o_bad | r_bad)[0]) if (o_bad | r_bad).any() else -1
                msgs.append(f"out{k}: nonfinite rows served {int(o_bad.sum())} ref {int(r_bad.sum())} "
                            f"(first {first}), rel err {rel:.2e}")
        if msgs or bad_in:
            state["events"] += 1
            amax = max((t.detach().abs().float().nan_to_num(0).max().item() for t in inputs
                        if isinstance(t, torch.Tensor) and t.is_floating_point()), default=0.0)
            log.warning("DSV4_MHC_CHECK call %d %s T=%d nonfinite inputs %s, max|input| %.3g: %s",
                        state["calls"], op, n, bad_in, amax, "; ".join(msgs) or "outputs ok")

    def wrap(cls, op, ref):
        orig = cls.forward_hip

        def forward_hip(self, *a, **k):
            out = orig(self, *a, **k)
            ref_out = ref(self, *a, **k)
            outs = out if isinstance(out, tuple) else (out,)
            refs = ref_out if isinstance(ref_out, tuple) else (ref_out,)
            inputs = list(a) + list(k.values())
            check(op, inputs, outs, refs)
            return out

        cls.forward_hip = forward_hip

    wrap(m.MHCPreOp, "pre", lambda self, *a, **k: self.forward_native(*a, **k))
    wrap(m.MHCPostOp, "post", lambda self, *a, **k: self.forward_native(*a, **k))
    wrap(m.MHCFusedPostPreOp, "fused", lambda self, *a, **k: self.forward_native(*a, **k))
    wrap(m.HCHeadOp, "head", lambda self, *a, **k: self.forward_xpu(*a, **k))


def install_native(ops: str) -> None:
    """DSV4_MHC_NATIVE=head,fusedpost,fused,pre: route the listed mHC ops back to reference
    implementations while the rest stays on the fused TileLang path (graph-safe bisection).

      head       hc_head via vLLM's Triton kernel instead of TileLang
      fusedpost  fused post+pre as torch post, then vLLM's mhc_pre_tilelang (TileLang prenorm
                 GEMM + the big-fuse kernel); drops TileLang's fused post+GEMM kernels
      fused      fused post+pre entirely in torch
      pre        standalone pre (first layer) in torch
    """
    import vllm.model_executor.layers.mhc as m
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    for op in filter(None, ops.split(",")):
        if op == "head":
            m.HCHeadOp.forward_hip = m.HCHeadOp.forward_xpu
        elif op == "fused":
            m.MHCFusedPostPreOp.forward_hip = m.MHCFusedPostPreOp.forward_native
        elif op == "pre":
            m.MHCPreOp.forward_hip = m.MHCPreOp.forward_native
        elif op == "fusedpost":
            def forward_hip(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                            hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                            sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                            norm_eps=0.0):
                residual_cur = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
                post, comb, layer_input = torch.ops.vllm.mhc_pre_tilelang(
                    residual_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, 1, norm_weight, norm_eps or 1e-6)
                return residual_cur, post, comb, layer_input

            m.MHCFusedPostPreOp.forward_hip = forward_hip
        else:
            raise ValueError(f"DSV4_MHC_NATIVE: unknown op {op!r}")
