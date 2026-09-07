#!/usr/bin/env python3
"""Measure collective bandwidth against message size, for one NCCL_MAX_NCHANNELS setting.

This exists to answer one question from the report: what does capping channels cost?
The reporter cares specifically about bandwidth-bound training, so all three collectives
that bound data-parallel training are measured (all_reduce for gradients, plus
all_gather and reduce_scatter for sharded optimisers), and the result is reported as
**bus bandwidth** as well as algorithm bandwidth so it can be compared with published
RCCL/nccl-tests figures.

rccl-tests is not installed anywhere under /appl on LUMI, and building it inside the
container would add a moving part for no gain: these are the same torch.distributed
entry points vLLM and DDP actually call, so measuring them measures the real thing.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist


def env_int(*names: str, default: int | None = None) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    if default is None:
        raise RuntimeError(f"none of {names} set")
    return default


def time_op(fn, reps: int, warmup: int) -> float:
    """Median seconds per call, warmup discarded.

    Median rather than mean: on a shared fabric one interfering neighbour produces an
    outlier that a mean would silently fold into the reported bandwidth.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        # Every rank must enter the collective before timing starts, or the measurement
        # includes skew between ranks rather than the collective itself.
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return statistics.median(samples)


def bus_factor(op: str, n: int) -> float:
    """nccl-tests bus-bandwidth conventions, so numbers are comparable to published ones."""
    if op == "all_reduce":
        return 2.0 * (n - 1) / n
    # all_gather and reduce_scatter each move (n-1)/n of the buffer across the fabric.
    return (n - 1) / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="default_channels")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--min-bytes", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter")
    args = parser.parse_args()

    rank = env_int("RANK", "SLURM_PROCID")
    world_size = env_int("WORLD_SIZE", "SLURM_NPROCS")
    local_rank = env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    device_index = 0 if torch.cuda.device_count() <= 1 else local_rank
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    dist.init_process_group(backend="nccl")

    rows = []
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    size = args.min_bytes
    while size <= args.max_bytes:
        for op in ops:
            elements = max(1, size // 4)  # float32
            try:
                if op == "all_reduce":
                    buf = torch.ones(elements, dtype=torch.float32, device=device)
                    fn = lambda b=buf: dist.all_reduce(b)
                    moved = size
                elif op == "all_gather":
                    # `size` is the total output, matching nccl-tests.
                    per_rank = max(1, elements // world_size)
                    src = torch.ones(per_rank, dtype=torch.float32, device=device)
                    dst = torch.empty(per_rank * world_size, dtype=torch.float32, device=device)
                    fn = lambda d=dst, s=src: dist.all_gather_into_tensor(d, s)
                    moved = per_rank * world_size * 4
                elif op == "reduce_scatter":
                    per_rank = max(1, elements // world_size)
                    src = torch.ones(per_rank * world_size, dtype=torch.float32, device=device)
                    dst = torch.empty(per_rank, dtype=torch.float32, device=device)
                    fn = lambda d=dst, s=src: dist.reduce_scatter_tensor(d, s)
                    moved = per_rank * world_size * 4
                else:
                    continue

                seconds = time_op(fn, args.reps, args.warmup)
                algbw = moved / seconds / 1e9
                rows.append({
                    "op": op,
                    "bytes": moved,
                    "seconds": seconds,
                    "algbw_gbps": round(algbw, 3),
                    "busbw_gbps": round(algbw * bus_factor(op, world_size), 3),
                })
                del fn
            except torch.cuda.OutOfMemoryError:
                # Record the ceiling rather than dying: a missing point is data too.
                rows.append({"op": op, "bytes": size, "error": "OOM"})
                torch.cuda.empty_cache()
            finally:
                torch.cuda.empty_cache()
        size *= 2

    if rank == 0:
        report = {
            "variant": args.variant,
            "world_size": world_size,
            "nodes": os.environ.get("SLURM_JOB_NUM_NODES", "?"),
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "host": socket.gethostname(),
            "reps": args.reps,
            "warmup": args.warmup,
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("NCCL_", "RCCL_", "FI_"))},
            "rows": rows,
        }
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, f"bandwidth_{args.variant}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"[{args.variant}] world_size={world_size}, wrote {out_path}")
        print(f"  {'op':16s} {'MiB':>10s} {'algbw GB/s':>12s} {'busbw GB/s':>12s}")
        for row in rows:
            if "error" in row:
                print(f"  {row['op']:16s} {row['bytes']/1048576:10.2f} {'OOM':>12s}")
            else:
                print(f"  {row['op']:16s} {row['bytes']/1048576:10.2f} "
                      f"{row['algbw_gbps']:12.2f} {row['busbw_gbps']:12.2f}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
