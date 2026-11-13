"""vLLM's custom all-reduce on gfx90a, with an L2 fence (DSV4_CUSTOM_AR=1).

vLLM 0.26 enables its custom all-reduce on ROCm only for gfx94x/95x. On gfx90a its kernels
are built and LUMI's 8 GCDs are all one XGMI hop apart, so lifting that gate makes TP
all-reduces 3-4.7x faster than RCCL at decode sizes (test_allreduce.py). But on its own it
gives garbage in serving (GSM8K 0 %, jobs 23671136/7): the one-stage kernel reads every
peer's input buffer over XGMI after a relaxed start barrier, and on gfx90a the L2 is per
GCD and not coherent across XGMI. The peer's input can still sit dirty in the peer's L2,
and the reader's L2 keeps remote lines from the previous step (graph buffers reuse the same
addresses). MI300 differs in its cache coherence design.

The fix: a one-thread kernel right before each graph-captured custom all-reduce, doing a
system-scope acq_rel atomic. On gfx90a that compiles to buffer_wbl2 (write back dirty L2
lines, so peers read this rank's input from memory) and buffer_invl2 (drop cached remote
lines). Eager calls go to RCCL instead: there the C++ op copies the input into its IPC
buffer and launches the kernel in one call, leaving no point for the fence.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _xgmi_fence_kernel(ptr):
    tl.atomic_add(ptr, 0, sem="acq_rel", scope="sys")


def xgmi_fence(buf: torch.Tensor) -> None:
    _xgmi_fence_kernel[(1,)](buf, num_warps=1)


def fence_asm(buf: torch.Tensor) -> str:
    """The fence kernel's gfx assembly (to check it has buffer_wbl2 and buffer_invl2)."""
    return _xgmi_fence_kernel[(1,)](buf, num_warps=1).asm["amdgcn"]


def install() -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.platforms.rocm import RocmPlatform

    # Read by ParallelConfig.__post_init__, after EngineArgs has loaded the plugins.
    RocmPlatform.use_custom_allreduce = classmethod(lambda cls: True)

    should = CustomAllreduce.should_custom_ar
    reduce = CustomAllreduce.all_reduce

    def should_custom_ar(self, inp):
        return self._IS_CAPTURING and should(self, inp)  # eager -> RCCL

    def all_reduce(self, inp, *, out=None, registered=False):
        if registered:
            buf = getattr(self, "_lumi_fence_buf", None)
            if buf is None:
                buf = self._lumi_fence_buf = torch.zeros(1, dtype=torch.int32, device=self.device)
            xgmi_fence(buf)
        return reduce(self, inp, out=out, registered=registered)

    CustomAllreduce.should_custom_ar = should_custom_ar
    CustomAllreduce.all_reduce = all_reduce
