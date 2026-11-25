"""lumi_dsv41 override of upstream's amd/mono_decode.py: the opt-in decode monokernel
(--monokernel, gfx942/gfx950, aiter FlyDSL) is not ported, so no layer ever gets one."""


class MonoDecodeLayer:
    @staticmethod
    def create(layer, vllm_config):
        """Upstream: record the flag the gfx942 decode helpers read, then build the
        layer only when kernel_config.monokernel is set."""
        from vllm.models.deepseek_v41.amd import ops

        ops.record_monokernel(vllm_config)
        if vllm_config.kernel_config.monokernel:
            raise NotImplementedError("the AMD decode monokernel is not ported to gfx90a")
        return None
