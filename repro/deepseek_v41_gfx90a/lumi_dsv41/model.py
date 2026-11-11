"""DeepSeek-V4.1 language model on vLLM 0.26: upstream's AMD class plus three adaptations.

- Dense MXFP8 linears (32x32 blocks, e8m0 scales) are loaded as block-FP8 through
  0.26's Fp8LinearMethod (quant.py), whose scale parameter is `weight_scale_inv`;
  upstream maps checkpoint scales to `weight_scale` for its ModelOpt MXFP8 method.
- Model Runner V2 state: upstream's (nvidia/model_state.py) also implements SWA bounded
  replay and decoder-replay CUDA graphs, which need engine support 0.26 lacks. Engram
  only needs each request's lookback token ids; that part is ported here.
- Engine config fields upstream reads that 0.26 does not have are set to their defaults.
"""

from typing import Any

import torch

from vllm.models.deepseek_v41.amd import model as upstream
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

def _set_missing_defaults(vllm_config) -> None:
    # Missing config fields are class-level defaults (hooks.CONFIG_DEFAULTS); the
    # Engram config is per instance.
    if getattr(vllm_config, "engram_config", None) is None:
        # Upstream VllmConfig._resolve_and_verify_engram_config, for a model with Engram.
        from vllm.config.engram import EngramConfig  # vendored, new since 0.26

        engram = EngramConfig()
        engram.verify_model_config(vllm_config.model_config)
        engram.resolve_dp_shared_memory(vllm_config.parallel_config)
        engram.verify_parallel_config(vllm_config.parallel_config)
        object.__setattr__(vllm_config, "engram_config", engram)


def _moe_fields() -> None:
    """Upstream's V4.1 MoE subclasses the ROCm V4 MoE and fills in the protocol fields an
    older base lacks; 0.26's base also lacks these two."""
    cls = upstream.DeepseekV4MoE
    if getattr(cls, "_lumi_dsv41", False):
        return
    init = cls.__init__

    def __init__(self, vllm_config, *args, **kwargs):
        init(self, vllm_config, *args, **kwargs)
        config = vllm_config.model_config.hf_config
        if not hasattr(self, "n_shared_experts"):
            self.n_shared_experts = config.n_shared_experts
        if not hasattr(self, "use_mega_moe"):
            self.use_mega_moe = False

    cls.__init__ = __init__
    cls._lumi_dsv41 = True


_moe_fields()


class DeepseekV41ForCausalLM(upstream.DeepseekV41LLMForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        _set_missing_defaults(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        expert_dtype = getattr(self.config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = upstream._make_deepseek_v4_weights_mapper(
            expert_dtype, "weight_scale_inv"
        )

    @staticmethod
    def get_model_state_cls():
        return DeepseekV41ModelState

    def load_weights(self, weights):
        # Language model only (vision_n_layers=0): the image-sentinel routing bias has
        # no parameter; upstream skips the other vision tensors itself.
        return super().load_weights(
            (name, tensor) for name, tensor in weights if not name.endswith(".gate.bias_vl")
        )

    def forward(self, *args, **kwargs):
        # load_weights() derives the mHC broadcast weights after loading; with
        # --load-format dummy it never runs, so derive them on first use.
        if not getattr(self, "_mhc_finalized", False):
            self.model.finalize_mhc_broadcast_weights()
            self._mhc_finalized = True
        return super().forward(*args, **kwargs)


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    positions_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # Verbatim from upstream nvidia/model_state.py: one program per lookback row;
    # rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    begin = tl.load(query_start_loc_ptr + batch_idx, mask=in_batch, other=0)
    end = tl.load(query_start_loc_ptr + batch_idx + 1, mask=in_batch, other=0)
    in_batch = in_batch & (end > begin)
    num_computed = tl.load(positions_ptr + begin, mask=in_batch, other=0)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx.to(tl.int64) * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the Engram lookback window (upstream's, minus replay).

    The n-gram hash needs the ids of the `depth` tokens before each request's chunk
    start. The runner keeps every request's token history on the device, so the window
    is gathered there each step, into a persistent buffer that CUDA graphs can read.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def prepare_inputs(self, input_batch, req_states) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
