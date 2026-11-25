"""vLLM's custom all-reduce on gfx90a, made coherent.

vLLM 0.26 enables its custom all-reduce on ROCm only for gfx94x/95x. On gfx90a the kernels
are built and the 8 GCDs of a LUMI-G node are one XGMI hop apart, so lifting the gate makes
tensor-parallel all-reduces 1.4-3.4x faster than RCCL at decode sizes. On its own it gives
garbage: the one-stage kernel reads each peer's input over XGMI, and on gfx90a the L2 is
per GCD and not coherent across XGMI, so the peer's input can still sit dirty in its L2.

Fix: always take the kernel's copy path, also inside CUDA graphs. The input is copied
into vLLM's IPC buffer, which ROCm builds allocate uncached, so no L2 holds it.
"""


def install() -> None:
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.platforms.rocm import RocmPlatform

    # Read by ParallelConfig.__post_init__, after EngineArgs has loaded the plugins.
    RocmPlatform.use_custom_allreduce = classmethod(lambda cls: True)
    reduce = CustomAllreduce.all_reduce

    def all_reduce(self, inp, *, out=None, registered=False):
        return reduce(self, inp, out=out, registered=False)  # via the uncached buffer

    CustomAllreduce.all_reduce = all_reduce
