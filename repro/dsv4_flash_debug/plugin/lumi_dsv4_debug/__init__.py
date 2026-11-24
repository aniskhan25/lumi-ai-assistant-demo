"""DeepSeek-V4 on gfx90a debugging switches for LAIF's vLLM 0.26 (vllm.general_plugins).

Each switch is an environment variable, so one variant runs per allocation:

DSV4_INDEXER_GATE=1  Lift 0.26's aiter-only gate on the sparse attention indexer and run
                     the op's non-aiter fallbacks (Triton cache insert/gather, compiled
                     top-k, torch FP8 logits), as in ../deepseek_v41_gfx90a/indexer_gate.
                     Registers the op's torch.ops entry, which 0.26 does only with aiter,
                     so it also works under torch.compile.
                     Also bounds the memory of the torch MQA-logits fallback that path
                     uses for prefill (see _chunked_mqa_logits).
DSV4_TRITON_PAGED_LOGITS=1
                     With the gate: compute the indexer's decode logits with one batched
                     Triton kernel (paged_logits.py) instead of the per-sequence torch loop,
                     which syncs with the host per sequence and cannot be graph-captured.
DSV4_BF16_LINEAR=1   Dequantise block-FP8 dense linears to bf16 once after loading and run
                     them with vLLM's ROCm bf16 GEMM (bf16_linear.py): on gfx90a 20-90x
                     faster than the W8A8 Triton kernel. Activations are still rounded to
                     FP8 first, which the model needs (DSV4_BF16_LINEAR_ACT=bf16 skips it).
DSV4_MHC_FIX=triton|wave64
                     Keep the fused TileLang mHC path but replace its wrong pre "big fuse"
                     kernel (mhc_fix.py): a Triton kernel, or the TileLang one at 128 threads
                     split at 64 (it splits a 96-thread block at 32, i.e. inside a wavefront).
                     Exclusive with DSV4_TORCH_MHC.
DSV4_MHC_FAST=1      With DSV4_MHC_FIX=triton: fused post+pre and pre at decode sizes as Triton
                     kernels (post, split-K prenorm GEMM, big fuse; mhc_fast.py, from the V4.1
                     port) instead of TileLang's post and one-split prenorm GEMM.
DSV4_MHC_CHECK=1     Debug, eager only: log mHC calls whose outputs are non-finite or differ from
                     the torch/Triton reference (mhc_fix.install_check).
DSV4_ROUTER_GUARD=1  nan_to_num the MoE gate logits before ROCm's top-k (router_guard.py):
                     non-finite graph pad rows otherwise get duplicate expert ids, which
                     corrupt the MoE routing of real tokens. =count also logs how many.
DSV4_SPARSE_SPLIT=1  Decode sparse attention with vLLM's split-K kernels, gated to gfx942/950,
                     instead of the one-workgroup-per-query fallback (sparse_split.py).
DSV4_SPARSE_DECODE=1 Decode sparse attention with the V4.1 port's split-K kernel at every
                     batch size (sparse_decode.py); supersedes DSV4_SPARSE_SPLIT.
DSV4_CUSTOM_AR=copy|fence (1 = copy)
                     Use vLLM's custom all-reduce for TP (0.26 enables it on ROCm only for
                     gfx94x/95x), made coherent on gfx90a: as is, ranks read stale peer data
                     from their L2 (custom_ar.py).
DSV4_SCORE_GEMM=1    Compressor score GEMMs (fp32 output) in a Triton kernel with bf16 MFMA;
                     hipBLASLt picks an fp32-MFMA kernel for them on gfx90a (score_gemm.py).
DSV4_MOE_SMALL=1     MoE at decode sizes: pack_bitmatrix with 16-row blocks, expert GEMM
                     tiles, warps and stages by token count (moe_small.py; DSV4_MOE_TABLE=v1
                     for the earlier block_n / split_k table).
DSV4_QNORM_ROPE=1    Attention's q-norm / RoPE / KV RoPE + FP8 cache insert in one Triton kernel
                     instead of vLLM's HIP op (qnorm_rope.py).
DSV4_MOE_GLUE=1      Fewer small kernels around the MoE: one routing-prologue kernel, int32 row
                     gathers, one shared-expert SiluAndMulWithClamp kernel (moe_glue.py).
DSV4_GATE_GEMM=1     MoE router GEMM (fp32 logits) via score_gemm.score_mm (moe_small.py).
DSV4_DENSE_GEMM=1    Dense bf16 GEMMs at decode sizes via score_gemm's Triton kernel (bf16 out)
                     where it beats vLLM's ROCm dispatch, per shape and size (dense_gemm.py).
DSV4_WO_A=1          Attention wo_a GEMM (rocm_inv_rope_einsum) via wvSplitK at 1-3 tokens and
                     score_gemm's Triton kernel with bf16 output at 4-128 (wo_a.py).
DSV4_GRAPH_DIFF=1    Debug: after each graph replay rerun the step eagerly and log the first
                     layer site where graph and eager differ (graph_diff.py).
DSV4_TORCH_MHC=1     Turn off the TileLang hyper-connection (mHC) kernels on gfx90a.
                     0.26 already does this for gfx942 ("TileLang MHC currently produces
                     incorrect results on gfx942", vllm/model_executor/layers/mhc.py), but
                     enables TileLang on every other ROCm GPU, gfx90a included. With it off,
                     mhc_pre/mhc_post use their torch fallbacks and hc_head its Triton one,
                     exactly as on gfx942, and the decoder layer takes the unfused path.
"""

import os
import sys


def _log(msg: str) -> None:
    from vllm.logger import init_logger

    init_logger("vllm.lumi_dsv4_debug").info(msg)


def _torch_mhc() -> None:
    import vllm.model_executor.layers.mhc as mhc

    mhc.HAS_TILELANG_MHC = False
    # The decoder layer copies the flag at import time to choose fused vs unfused.
    model = sys.modules.get("vllm.models.deepseek_v4.amd.model")
    if model is not None:
        model.HAS_TILELANG_MHC = False
    _log("DSV4_TORCH_MHC: TileLang mHC off; torch/triton fallbacks as on gfx942")


def _chunked_mqa_logits() -> None:
    """0.26's fp8_mqa_logits_torch materialises a dense fp32 [heads, rows, kv] score in
    one go: 4 GiB for an 8192-token prefill chunk at 18k context, which OOMs next to the
    KV cache (job 23494525). Rows are independent, so compute them in chunks capped at
    ~512 MiB of scores: the same arithmetic in the same order per row."""
    import torch

    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ops

    stock = ops.fp8_mqa_logits_torch
    if getattr(stock, "_lumi_chunked", False):
        return

    def fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        m, h = q.shape[0], q.shape[1]
        n = kv[0].shape[0]
        rows = max(1, (512 << 20) // max(1, h * n * 4))
        if rows >= m:
            return stock(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
        out = torch.empty(m, n, dtype=torch.float32, device=q.device)
        for start in range(0, m, rows):
            end = min(m, start + rows)
            out[start:end] = stock(
                q[start:end], kv, weights[start:end],
                cu_seqlen_ks[start:end], cu_seqlen_ke[start:end],
            )
        return out

    fp8_mqa_logits_torch._lumi_chunked = True
    ops.fp8_mqa_logits_torch = fp8_mqa_logits_torch


def _register_indexer_op() -> None:
    """0.26 registers torch.ops.vllm.rocm_aiter_sparse_attn_indexer only when aiter is on
    (vllm/_aiter_ops.py), although the op itself has non-aiter fallbacks and is already a
    graph-splitting op (CompilationConfig._attention_ops). Register it the same way, so
    the indexer runs as one opaque op under torch.compile instead of being traced."""
    import torch

    from vllm._aiter_ops import rocm_aiter_ops

    if rocm_aiter_ops.is_enabled() or hasattr(torch.ops.vllm, "rocm_aiter_sparse_attn_indexer"):
        return
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_aiter_sparse_attn_indexer,
        rocm_aiter_sparse_attn_indexer_fake,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_sparse_attn_indexer",
        op_func=rocm_aiter_sparse_attn_indexer,
        mutates_args=["topk_indices_buffer"],
        fake_impl=rocm_aiter_sparse_attn_indexer_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def _indexer_gate() -> None:
    import torch

    from vllm.model_executor.layers import sparse_attn_indexer as module

    cls = module.SparseAttnIndexer
    stock = cls.forward_hip
    if getattr(stock, "_lumi_gate", False):
        return

    def forward_hip(self, hidden_states, q_quant, k, weights):
        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_enabled():
            return stock(self, hidden_states, q_quant, k, weights)
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), "expects a single FP8 q_quant tensor"
        # Through torch.ops (registered by _register_indexer_op): torch.compile splits the
        # graph at it, as at vllm::rocm_aiter_sparse_attn_indexer with aiter.
        return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
            hidden_states,
            module._encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=self.skip_k_cache_insert,
        )

    forward_hip._lumi_gate = True
    cls.forward_hip = forward_hip
    _register_indexer_op()
    _chunked_mqa_logits()
    _log("DSV4_INDEXER_GATE: sparse indexer runs its non-aiter fallbacks (chunked MQA logits)")


_STACK_FILE = None

def register() -> None:
    import faulthandler
    import signal

    # `pkill -USR1 -f VLLM::Worker` dumps every worker's Python stacks into the server log.
    # DSV4_STACK_DIR: one file per process, since worker stderr goes through a pipe that vLLM
    # prefixes and forwards, where dumps from a hung process were lost.
    stack_dir = os.environ.get("DSV4_STACK_DIR")
    if stack_dir:
        os.makedirs(stack_dir, exist_ok=True)
        global _STACK_FILE
        _STACK_FILE = open(os.path.join(stack_dir, f"stack-{os.getpid()}.txt"), "w")
        faulthandler.register(signal.SIGUSR1, file=_STACK_FILE, all_threads=True)
    else:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    car = os.environ.get("DSV4_CUSTOM_AR", "0")
    if car != "0":
        from . import custom_ar

        custom_ar.install("copy" if car == "1" else car)
        _log(f"DSV4_CUSTOM_AR={car}: vLLM custom all-reduce on gfx90a (custom_ar.py)")
    mhc_mode = os.environ.get("DSV4_MHC_FIX", "")
    if os.environ.get("DSV4_TORCH_MHC") == "1":
        if mhc_mode:
            raise ValueError("DSV4_TORCH_MHC=1 and DSV4_MHC_FIX are exclusive")
        _torch_mhc()
    elif mhc_mode:
        from . import mhc_fix

        mhc_fix.install(mhc_mode)
        _log(f"DSV4_MHC_FIX={mhc_mode}: fused TileLang mHC with a corrected pre big-fuse kernel")
    if os.environ.get("DSV4_MHC_FAST") == "1":
        if mhc_mode != "triton":
            raise ValueError("DSV4_MHC_FAST=1 needs DSV4_MHC_FIX=triton")
        from . import mhc_fast

        mhc_fast.install()
        _log(f"DSV4_MHC_FAST=1: mHC post + split-K prenorm GEMM + big fuse ({mhc_fast.BF_KERNEL}) in Triton at decode sizes")
    if os.environ.get("DSV4_MHC_NATIVE"):
        from . import mhc_fix

        mhc_fix.install_native(os.environ["DSV4_MHC_NATIVE"])
        _log(f"DSV4_MHC_NATIVE={os.environ['DSV4_MHC_NATIVE']}: these mHC ops use reference kernels")
    if os.environ.get("DSV4_MHC_META") == "1":
        from . import mhc_fix

        mhc_fix.install_meta()
        _log("DSV4_MHC_META: logging mHC argument layouts")
    if os.environ.get("DSV4_NANFILL") == "1":
        from . import mhc_fix

        mhc_fix.install_nanfill()
        _log("DSV4_NANFILL: fresh float CUDA allocations filled with NaN")
    if os.environ.get("DSV4_MHC_NANTRACE") == "1":
        from . import mhc_fix

        mhc_fix.install_nantrace()
        _log("DSV4_MHC_NANTRACE: per-row non-finite flags at every mHC site, read per step")
    if os.environ.get("DSV4_MHC_CHECK") == "1":
        from . import mhc_fix

        mhc_fix.install_check()
        _log("DSV4_MHC_CHECK: every mHC call recomputed with the reference and compared (eager only)")
    if os.environ.get("DSV4_SPARSE_DECODE") == "1":
        if os.environ.get("DSV4_SPARSE_SPLIT") == "1":
            raise ValueError("DSV4_SPARSE_DECODE=1 supersedes DSV4_SPARSE_SPLIT=1; set one")
        from . import sparse_decode

        sparse_decode.install()
        _log("DSV4_SPARSE_DECODE=1: sparse attention decode with the V4.1 split-K kernel")
    if os.environ.get("DSV4_SPARSE_SPLIT") == "1":
        from . import sparse_split

        sparse_split.install()
        _log("DSV4_SPARSE_SPLIT=1: split-K sparse attention decode (gfx942/950 path)")
    if os.environ.get("DSV4_ROUTER_GUARD") in ("1", "count"):
        from . import router_guard

        router_guard.install()
        _log(f"DSV4_ROUTER_GUARD={os.environ['DSV4_ROUTER_GUARD']}: non-finite gate rows sanitised before top-k")
    if os.environ.get("DSV4_GRAPH_DIFF") == "1":
        from . import graph_diff

        graph_diff.install()
        _log("DSV4_GRAPH_DIFF: every graph replay rerun eagerly and compared per layer")
    if os.environ.get("DSV4_SCORE_GEMM") == "1":
        from . import score_gemm

        score_gemm.install()
        _log("DSV4_SCORE_GEMM=1: compressor score GEMMs in Triton (bf16 MFMA, fp32 out)")
    if os.environ.get("DSV4_WO_A") == "1":
        from . import wo_a

        wo_a.install()
        _log("DSV4_WO_A=1: attention wo_a GEMM via wvSplitK / Triton instead of einsum")
    if os.environ.get("DSV4_QNORM_ROPE") == "1":
        from . import qnorm_rope

        qnorm_rope.install()
        _log("DSV4_QNORM_ROPE=1: q-norm / RoPE / KV insert in one Triton kernel")
    if os.environ.get("DSV4_MOE_GLUE") == "1":
        from . import moe_glue

        moe_glue.install()
        _log("DSV4_MOE_GLUE=1: one-kernel MoE routing prologue, int32 row gathers, fused shared-expert activation")
    if os.environ.get("DSV4_MOE_SMALL") == "1":
        from . import moe_small

        moe_small.install_pack()
        moe_small.install_matmul()
        _log("DSV4_MOE_SMALL=1: pack_bitmatrix row blocks of 16, expert GEMM tiles by token count")
    if os.environ.get("DSV4_GATE_GEMM") == "1":
        from . import moe_small

        moe_small.install_gate()
        _log("DSV4_GATE_GEMM=1: MoE router GEMM via score_mm (fp32 logits)")
    if os.environ.get("DSV4_DENSE_GEMM") == "1":  # before BF16_LINEAR binds the GEMM
        from . import dense_gemm

        dense_gemm.install()
        _log("DSV4_DENSE_GEMM=1: dense bf16 GEMMs at decode sizes via Triton where faster")
    if os.environ.get("DSV4_BF16_LINEAR") == "1":
        from . import bf16_linear

        bf16_linear.install()
        _log("DSV4_BF16_LINEAR: block-FP8 dense linears will run in bf16")
    if os.environ.get("DSV4_INDEXER_GATE") == "1":
        _indexer_gate()
        if os.environ.get("DSV4_TRITON_PAGED_LOGITS") == "1":
            from . import paged_logits

            paged_logits.install()
            _log("DSV4_TRITON_PAGED_LOGITS: indexer decode logits via one batched Triton kernel")
