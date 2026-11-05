"""DeepSeek-V4 on gfx90a debugging switches for LAIF's vLLM 0.26 (vllm.general_plugins).

Each switch is an environment variable, so one variant runs per allocation:

DSV4_INDEXER_GATE=1  Lift 0.26's aiter-only gate on the sparse attention indexer and call
                     the op's Python implementation (Triton cache insert/gather, compiled
                     top-k, torch FP8 logits), as in ../deepseek_v41_gfx90a/indexer_gate.
                     Also bounds the memory of the torch MQA-logits fallback that path
                     uses for prefill (see _chunked_mqa_logits).
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
        # The torch.ops entry is registered only with aiter; call the Python op (eager).
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            rocm_aiter_sparse_attn_indexer,
        )

        return rocm_aiter_sparse_attn_indexer(
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
    _chunked_mqa_logits()
    _log("DSV4_INDEXER_GATE: sparse indexer runs its non-aiter fallbacks (chunked MQA logits)")


def register() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    if os.environ.get("DSV4_TORCH_MHC") == "1":
        _torch_mhc()
    if os.environ.get("DSV4_INDEXER_GATE") == "1":
        _indexer_gate()
