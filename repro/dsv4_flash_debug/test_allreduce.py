"""vLLM's custom all-reduce vs RCCL on 8 MI250X GCDs at decode sizes (torchrun, 8 ranks).

vLLM 0.26 enables its custom all-reduce on ROCm only for gfx94x/95x, and only when every GPU
pair is one XGMI hop apart, so at TP=8 on LUMI every all-reduce goes through RCCL. The
kernels are built into this gfx90a container (_C_stable_libtorch). This prints the XGMI
hop matrix, forces the custom all-reduce on, checks its result against RCCL and times both
as served: CUDA-graph replays of out-of-place bf16 all-reduces of [tokens, 4096].

    torchrun --nproc-per-node 8 test_allreduce.py
"""

import os

import torch
import torch.distributed as dist

from vllm.platforms import current_platform

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("gloo")
cpu_group = dist.new_group(backend="gloo")

if rank == 0:
    from amdsmi import (amdsmi_get_processor_handles, amdsmi_init, amdsmi_shut_down,
                        amdsmi_topo_get_link_type)
    amdsmi_init()
    h = amdsmi_get_processor_handles()
    print("XGMI hops (type 2 = XGMI):")
    for i in range(world):
        row = []
        for j in range(world):
            if i == j:
                row.append("  -")
                continue
            t = amdsmi_topo_get_link_type(h[i], h[j])
            row.append(f"{t['hops']:>2}{'x' if t['type'] == 2 else '?'}")
        print(f"  GCD{i}: " + " ".join(row))
    amdsmi_shut_down()
    print("vLLM is_fully_connected:", current_platform.is_fully_connected(list(range(world))))

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce  # noqa: E402
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

nccl = PyNcclCommunicator(group=cpu_group, device=dev)
type(current_platform).is_fully_connected = classmethod(lambda cls, ids: True)
ca = CustomAllreduce(group=cpu_group, device=dev)
assert not ca.disabled, "custom all-reduce did not initialise"

HIDDEN, ITERS = 4096, 20
SIZES = (1, 4, 8, 16, 32, 64, 128)


def graph_time(fn, x, capture_ctx=None):
    """us per call: replays of a graph holding ITERS calls (as served)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    if capture_ctx is not None:
        with capture_ctx():
            with torch.cuda.graph(g):
                for _ in range(ITERS):
                    out = fn(x)
    else:
        with torch.cuda.graph(g):
            for _ in range(ITERS):
                out = fn(x)
    times = []
    for _ in range(3):  # best of 3, each 10 replays; barrier so ranks start together
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / (10 * ITERS) * 1e3)
    return min(times), out


if rank == 0:
    print(f"\n{'tokens':>6} {'KiB':>6} {'max abs diff':>13} {'RCCL us':>8} {'custom us':>10} {'speedup':>8}")
ok = True
for t in SIZES:
    torch.manual_seed(rank * 1000 + t)
    x = torch.randn(t, HIDDEN, device=dev).bfloat16()
    ref = nccl.all_reduce(x)
    got = ca.custom_all_reduce(x)  # eager: copies into the registered buffer
    torch.cuda.synchronize()
    exact = torch.zeros(t, HIDDEN, device=dev)
    for r in range(world):  # fp32 reference from every rank's seed
        torch.manual_seed(r * 1000 + t)
        exact += torch.randn(t, HIDDEN, device=dev).bfloat16().float()
    d_ref = (ref.float() - exact).abs().max().item()
    d_ca = (got.float() - exact).abs().max().item()
    good = torch.isfinite(got).all().item() and d_ca <= max(2 * d_ref, 0.0625)
    t_nccl, _ = graph_time(nccl.all_reduce, x)
    t_ca, out = graph_time(ca.custom_all_reduce, x, ca.capture)
    torch.cuda.synchronize()
    good &= torch.equal(out, ca.custom_all_reduce(x))  # graph replay result = eager result
    flag = torch.tensor([int(good)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    ok &= bool(flag.item())
    if rank == 0:
        print(f"{t:>6} {t * HIDDEN * 2 // 1024:>6} {d_ca:>6.3f}/{d_ref:<6.3f} {t_nccl:>8.1f} "
              f"{t_ca:>10.1f} {t_nccl / t_ca:>7.2f}x{'' if flag.item() else '  MISMATCH'}")
if rank == 0:
    print("PASS" if ok else "FAIL")
ca.close()
dist.destroy_process_group()
