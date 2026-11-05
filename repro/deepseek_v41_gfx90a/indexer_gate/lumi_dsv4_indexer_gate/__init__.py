"""Phase-0 experiment, not a fix: lift vLLM 0.26's aiter-only gate on the DeepSeek-V4
sparse attention indexer, so gfx90a runs the op's own non-aiter fallbacks.

0.26's SparseAttnIndexer.forward_hip calls torch.ops.vllm.rocm_aiter_sparse_attn_indexer
only when rocm_aiter_ops.is_enabled(), and raises otherwise. Inside that op, everything
has a path that needs no aiter package: Triton cache insert and gather, vLLM's compiled
top_k_per_row, and torch fallbacks for the FP8 MQA logits when aiter is off. Upstream main
keeps the gate but also opens it for gfx950, noting that those fallbacks "do not require
the aiter package". This plugin opens it on any ROCm GPU and calls the op with exactly the
arguments 0.26 passes.
"""


def register() -> None:
    import torch

    from vllm.model_executor.layers import sparse_attn_indexer as module
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
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
        # 0.26 registers the torch.ops wrapper only when aiter is enabled
        # (_aiter_ops.py), so call the Python implementation directly. Fine in eager
        # mode; torch.compile would need the op registered.
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
