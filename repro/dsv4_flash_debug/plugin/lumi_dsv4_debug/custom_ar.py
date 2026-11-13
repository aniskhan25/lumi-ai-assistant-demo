"""vLLM's custom all-reduce on gfx90a, with an L2 fence (DSV4_CUSTOM_AR=1).

vLLM 0.26 enables its custom all-reduce on ROCm only for gfx94x/95x. On gfx90a its kernels
are built and LUMI's 8 GCDs are all one XGMI hop apart, so lifting that gate makes TP
all-reduces 3-4.7x faster than RCCL at decode sizes (test_allreduce.py). But on its own it
gives garbage in serving (GSM8K 0 %, jobs 23671136/7): the one-stage kernel reads every
peer's input buffer over XGMI after a relaxed start barrier, and on gfx90a the L2 is per
GCD and not coherent across XGMI. The peer's input can still sit dirty in the peer's L2,
and the reader's L2 keeps remote lines from the previous step (graph buffers reuse the same
addresses). MI300 differs in its cache coherence design.

Two fixes (DSV4_CUSTOM_AR=copy|fence):

copy   Always take the kernel's eager path, also in graphs: copy the input into vLLM's IPC
       buffer, which ROCm builds allocate uncached ("for signal on MI200"), so neither the
       writer's nor the reader's L2 holds it. Costs a device copy of at most max_size.
fence  Keep reading the caller's (cached) tensor in graphs, behind a one-thread kernel doing
       a system-scope acq_rel atomic, which on gfx90a compiles to buffer_wbl2 (write back
       dirty L2 lines) and buffer_invl2 (drop cached remote lines). Eager calls go to RCCL.
       Correct (GSM8K 97.2 %, jobs 23696206/7), but writing back a serving L2 full of
       fresh activations costs about half of the gain.
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


def install(mode: str = "copy") -> None:
    if mode not in ("copy", "fence"):
        raise ValueError(f"DSV4_CUSTOM_AR={mode!r}: expected copy or fence")
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.platforms.rocm import RocmPlatform

    # Read by ParallelConfig.__post_init__, after EngineArgs has loaded the plugins.
    RocmPlatform.use_custom_allreduce = classmethod(lambda cls: True)

    should = CustomAllreduce.should_custom_ar
    reduce = CustomAllreduce.all_reduce

    if mode == "copy":
        def copy_all_reduce(self, inp, *, out=None, registered=False):
            return reduce(self, inp, out=out, registered=False)  # via the uncached buffer

        CustomAllreduce.all_reduce = copy_all_reduce
        return

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
