"""Guard the MoE router against non-finite rows.

Under CUDA graphs a batch is padded to the captured size, and the pad rows carry whatever
the graph pool held, sometimes NaN/Inf. Every op but the MoE routing is row-local, so pad
rows are harmless elsewhere. ROCm's router kernel returns ids [0]*6 for an all-NaN row;
the triton_unfused experts build their dispatch/combine maps from a bitmatrix in which
those six slots set one bit, so the maps no longer line up and real tokens get the wrong
expert outputs.

nan_to_num on the gate logits leaves finite rows bit-identical and gives non-finite rows
six distinct ids. In place: no new allocation, so the graph-pool layout is unchanged.
"""

import torch


def install() -> None:
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as r

    orig = r.vllm_topk_softplus_sqrt

    def guarded(topk_weights, topk_indices, token_expert_indices, gating_output, *a, **k):
        torch.nan_to_num_(gating_output, nan=0.0)
        return orig(topk_weights, topk_indices, token_expert_indices, gating_output, *a, **k)

    r.vllm_topk_softplus_sqrt = guarded
