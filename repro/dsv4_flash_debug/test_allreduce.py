"""vLLM's custom all-reduce vs RCCL on 8 MI250X GCDs at decode sizes (torchrun, 8 ranks).

vLLM 0.26 enables its custom all-reduce on ROCm only for gfx94x/95x. Its kernels are built
into the gfx90a container and LUMI's 8 GCDs are all one XGMI hop apart. Forced on, it gave
garbage in serving although a test that reduced the same input every time passed: on
gfx90a the L2 is per GCD and not coherent across XGMI, so stale data looks right when the
data never changes. Here every replay first writes new data into the input with a kernel,
as a model layer does, and four variants are checked against the exact sum:

    RCCL     vLLM's PyNccl, as served today
    custom   the custom all-reduce reading the caller's tensor, as vLLM does in graphs
    fence    custom_ar.py's L2 writeback/invalidate fence first (DSV4_CUSTOM_AR=fence)
    copy     input copied into vLLM's uncached IPC buffer first (DSV4_CUSTOM_AR=copy)

then timed as served: CUDA-graph replays of bf16 all-reduces of [tokens, 4096], each after
a kernel that writes 16 MB (2x the L2), since the fence's write-back cost depends on how
much dirty data the L2 holds; the time of that kernel alone is subtracted.

    torchrun --nproc-per-node 8 test_allreduce.py
"""

import os
import sys

import torch
import torch.distributed as dist

from vllm.platforms import current_platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
from lumi_dsv4_debug.custom_ar import fence_asm, xgmi_fence  # noqa: E402

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("gloo")
cpu_group = dist.new_group(backend="gloo")
fence_buf = torch.zeros(1, dtype=torch.int32, device=dev)

if rank == 0:
    print("vLLM is_fully_connected (all pairs 1 XGMI hop):",
          current_platform.is_fully_connected(list(range(world))))
    ops = [ln.strip() for ln in fence_asm(fence_buf).splitlines()
           if "buffer_" in ln or "global_atomic" in ln]
    print("fence kernel:", "; ".join(ops))

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402

HIDDEN_ENV = int(os.environ.get("AR_HIDDEN", 4096))
SIZES_ENV = [int(x) for x in os.environ.get("AR_TOKENS", "1 4 16 32 64 128").split()]
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

nccl = PyNcclCommunicator(group=cpu_group, device=dev)
type(current_platform).is_fully_connected = classmethod(lambda cls, ids: True)
ca = CustomAllreduce(group=cpu_group, device=dev,
                     max_size=max(8 << 20, max(SIZES_ENV) * HIDDEN_ENV * 2 + (1 << 20)))
assert not ca.disabled, "custom all-reduce did not initialise"
xgmi_fence(fence_buf)  # compile before any capture

# AR_HIDDEN / AR_TOKENS: V4.1 is 5120 wide, and DSpark's verify batches reach 96-768 tokens.
HIDDEN, ITERS, STEPS = int(os.environ.get("AR_HIDDEN", 4096)), 20, 8
SIZES = tuple(int(x) for x in os.environ.get("AR_TOKENS", "1 4 16 32 64 128").split())


def rccl(x):
    return nccl.all_reduce(x)


def custom(x):
    return ca.all_reduce(x, registered=True)


def fenced(x):
    xgmi_fence(fence_buf)
    return ca.all_reduce(x, registered=True)


def copied(x):
    return ca.all_reduce(x, registered=False)


VARIANTS = {"RCCL": rccl, "custom": custom, "fence": fenced, "copy": copied}
junk = torch.zeros(16 << 20, device=dev, dtype=torch.bfloat16)  # 32 MB, > the largest message


def none(x):
    return x


def capture(calls):
    """Graph of `calls()`; ca.capture() registers the custom kernel's buffers at the end."""
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            out = calls()
    return g, out


def data(r, k, t):
    gen = torch.Generator(device=dev).manual_seed(r * 100003 + k * 101 + t)
    return torch.randn(t, HIDDEN, device=dev, generator=gen).bfloat16()


def max_err(fn, t):
    """Max |result - exact sum| over STEPS replays, each on freshly written data."""
    src = torch.zeros(t, HIDDEN, device=dev, dtype=torch.bfloat16)
    x = torch.zeros_like(src)
    g, out = capture(lambda: (x.copy_(src), fn(x))[1])
    err = 0.0
    for k in range(STEPS):
        src.copy_(data(rank, k, t))  # a kernel write, as a layer's output would be
        dist.barrier(group=cpu_group)
        g.replay()
        torch.cuda.synchronize()
        exact = sum(data(r, k, t).float() for r in range(world))
        err = max(err, (out.float() - exact).abs().max().item())
    e = torch.tensor([err])
    dist.all_reduce(e, op=dist.ReduceOp.MAX, group=cpu_group)
    return e.item()


def graph_time(fn, t):
    """us per call: replays of a graph holding ITERS calls (as served)."""
    x = torch.randn(t, HIDDEN, device=dev).bfloat16()
    g, _ = capture(lambda: [(junk.add_(1), fn(x))[1] for _ in range(ITERS)][-1])
    times = []
    for _ in range(3):
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / (10 * ITERS) * 1e3)
    return min(times)


# Warm RCCL up outside capture (communicator setup).
nccl.all_reduce(torch.zeros(8, HIDDEN, device=dev, dtype=torch.bfloat16))
torch.cuda.synchronize()

if rank == 0:
    print(f"\nmax |error| vs exact fp32 sum over {STEPS} replays on new data; "
          "GPU us per call after a 16 MB write")
    print(f"{'tokens':>6} {'KiB':>5}" + "".join(f"{n + ' err':>14}" for n in VARIANTS)
          + "".join(f"{n + ' us':>10}" for n in VARIANTS))
ok = True
for t in SIZES:
    errs = {n: max_err(fn, t) for n, fn in VARIANTS.items()}
    base = graph_time(none, t)
    times = {n: graph_time(fn, t) - base for n, fn in VARIANTS.items()}
    tol = max(2 * errs["RCCL"], 0.125)
    ok &= errs["fence"] <= tol and errs["copy"] <= tol
    if rank == 0:
        print(f"{t:>6} {t * HIDDEN * 2 // 1024:>5}"
              + "".join(f"{errs[n]:>10.3f}{'' if errs[n] <= tol else ' BAD':>4}" for n in VARIANTS)
              + "".join(f"{times[n]:>10.1f}" for n in VARIANTS))
if rank == 0:
    print("PASS (fence and copy match the exact sum)" if ok else "FAIL")
ca.close()
dist.destroy_process_group()
