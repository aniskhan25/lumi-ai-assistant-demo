"""Guard the V4 MoE router against non-finite rows (DSV4_ROUTER_GUARD).

Under CUDA graphs a batch is padded to the captured size, and the pad rows carry whatever
the graph pool held, sometimes NaN/Inf. Every op but the MoE routing is row-local, so pad
rows are harmless elsewhere. ROCm's router kernel (ops.topk_hash_softplus_sqrt) returns
ids [0]*6 for an all-NaN row (test_topk_nan.py); the triton_unfused experts build their
dispatch/combine maps from a bitmatrix in which those six slots set one bit, so the maps
no longer line up and real tokens get the wrong expert outputs (test_moe_pad_rows.py).

nan_to_num on the gate logits leaves finite rows bit-identical and gives non-finite rows
six distinct ids. In place: no new allocation, so the graph-pool layout is unchanged.

DSV4_ROUTER_GUARD=count also counts non-finite router rows and logs the running total
every 256 forward passes (debug; adds kernels).
"""

import os
import sys

import torch

_count: torch.Tensor | None = None
_steps = 0
_last = -1


def install() -> None:
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as r

    orig = r.vllm_topk_softplus_sqrt
    counting = os.environ.get("DSV4_ROUTER_GUARD") == "count"

    def guarded(topk_weights, topk_indices, token_expert_indices, gating_output, *a, **k):
        global _count
        if counting:
            if _count is None:
                _count = torch.zeros((), dtype=torch.int64, device=gating_output.device)
            _count += (~torch.isfinite(gating_output)).any(-1).sum()
        torch.nan_to_num_(gating_output, nan=0.0)
        return orig(topk_weights, topk_indices, token_expert_indices, gating_output, *a, **k)

    r.vllm_topk_softplus_sqrt = guarded
    if counting:
        _install_report()


def _install_report() -> None:
    import vllm.models.deepseek_v4.amd.model as m

    cls = m.DeepseekV4ForCausalLM
    orig = cls.compute_logits

    def compute_logits(self, *a, **k):
        global _steps, _last
        _steps += 1
        if _count is not None and _steps % 256 == 0:
            n = int(_count.item())
            if n != _last:
                _last = n
                print(f"[lumi_dsv4_debug] router: {n} non-finite gate rows after "
                      f"{_steps} forward passes", file=sys.stderr, flush=True)
        return orig(self, *a, **k)

    cls.compute_logits = compute_logits
