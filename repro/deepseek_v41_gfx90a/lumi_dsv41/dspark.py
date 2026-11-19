"""DeepSeek-V4.1's DSpark drafter (`mtp.{0,1,2}.*` in the target checkpoint) on vLLM 0.26.

0.26's Model Runner V2 already has the DSpark speculator (v1/worker/gpu/spec_decode/dspark)
and, for `--speculative-config '{"method": "dspark"}'` with a DeepSeek target, names the
draft architecture `DSparkDraftModel`; __init__.register maps that name here. The class is
upstream's AMD drafter with the target wrapper's adaptations (model.py):

- engine config fields 0.26 lacks get their defaults;
- dense MXFP8 linears load through 0.26's Fp8LinearMethod (quant.py), whose scale
  parameter is `weight_scale_inv`, not the `weight_scale` upstream picks for MXFP8;
- the vision routing bias (`gate.bias_vl`) is skipped at load;
- no confidence head: upstream uses it for adaptive verification, which 0.26's speculator
  does not do (it never calls compute_confidence), and 0.26 lacks the head's class.
"""

from vllm.models.deepseek_v41.amd import dspark as upstream

from .model import _set_missing_defaults


class DSparkDeepseekV41ForCausalLM(upstream.DSparkDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        _set_missing_defaults(vllm_config)
        draft_hf = vllm_config.speculative_config.draft_model_config.hf_config
        draft_hf.enable_confidence_head = False
        draft_hf.vision_n_layers = 0  # language model only, as for the target
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.linear_scale_name = "weight_scale_inv"

    def load_weights(self, weights):
        # As the target (model.py): language model only, so the image-sentinel routing
        # bias has no parameter (job 23861528: KeyError model.layers.0.ffn.gate.bias_vl).
        return super().load_weights(
            (name, tensor) for name, tensor in weights if not name.endswith(".gate.bias_vl")
        )
