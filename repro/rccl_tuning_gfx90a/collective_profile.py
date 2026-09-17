#!/usr/bin/env python3
"""Profile RCCL collectives for one slot of one block, and record enough to
prove afterwards that the slot measured what it claims to have measured.

Descends from repro/rccl_startup_gfx90a/bandwidth_sweep.py. Four things changed,
each because the ancestor answers a different question than this study asks:

  * **bf16, not fp32.** vLLM's tensor-parallel all-reduce is bf16. At a fixed byte
    count fp32 performs half the element reductions, which misattributes the
    reduce-kernel share of the time. A few fp32 sizes are kept so the numbers stay
    comparable with job 21791400.
  * **all_to_all_single and broadcast added.** The shipped serving config runs
    --enable-expert-parallel, so MoE all-to-all is a first-class collective here and
    behaves nothing like all-reduce on a Dragonfly. The ancestor never measured it.
  * **Latency by burst for small messages.** The ancestor barriers before every
    timed rep. The barrier sits outside the event bracket, so each rep is charged
    ~5-10 us of launch latency -- around 30% of the number at 8 B -- and barrier exit
    skew moves with the very knobs under test. Small sizes are additionally timed as
    a burst of K back-to-back calls after a single barrier, which is both the
    nccl-tests convention and what a decode loop actually experiences.
  * **Raw samples, every rank.** Without the samples there is no confidence
    interval, no p95 and no way to see a bimodal variant. The ancestor keeps only
    rank 0's median.

It also records the exact serving message sizes as explicit grid points, because
S_decode is not a power of two and interpolating a doubling ladder across it would
invent the one number the promotion scalar is built on.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time

# torch is imported inside main(), after --dry-run has had its chance to return:
# the size grid is the part worth checking on a laptop, and it needs no GPU stack.
torch = None
dist = None

# Latency regime: reported in microseconds, and additionally burst-timed.
LATENCY_MAX = 1024 * 1024
# all_to_all allocates world_size x the per-rank chunk; at world 64 a 2 GiB
# nominal size is 128 GiB of buffer. Cap it and record the ceiling.
A2A_MAX = 256 * 1024**2

BUS_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
}


def env_int(*names: str, default: int | None = None) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    if default is None:
        raise RuntimeError(f"none of {names} set")
    return default


def size_grid(min_bytes: int, max_bytes: int, extra: list[int]) -> list[int]:
    """Doubling ladder plus the exact serving sizes, which are not powers of two."""
    sizes, size = [], min_bytes
    while size <= max_bytes:
        sizes.append(size)
        size *= 2
    sizes.extend(s for s in extra if min_bytes <= s <= max_bytes)
    return sorted(set(sizes))


def make_op(op: str, nbytes: int, dtype: torch.dtype, device: str, world: int):
    """Return (callable, bytes_moved) or None if the size cannot be formed.

    `nbytes` follows the nccl-tests convention: for all_gather and reduce_scatter
    it is the size of the *whole* buffer, not the per-rank slice.
    """
    width = torch.finfo(dtype).bits // 8
    elements = max(1, nbytes // width)
    per_rank = max(1, elements // world)

    if op == "all_reduce":
        buf = torch.ones(elements, dtype=dtype, device=device)
        return (lambda: dist.all_reduce(buf)), elements * width
    if op == "all_gather":
        src = torch.ones(per_rank, dtype=dtype, device=device)
        dst = torch.empty(per_rank * world, dtype=dtype, device=device)
        return (lambda: dist.all_gather_into_tensor(dst, src)), per_rank * world * width
    if op == "reduce_scatter":
        src = torch.ones(per_rank * world, dtype=dtype, device=device)
        dst = torch.empty(per_rank, dtype=dtype, device=device)
        return (lambda: dist.reduce_scatter_tensor(dst, src)), per_rank * world * width
    if op == "all_to_all":
        if nbytes > A2A_MAX:
            return None
        src = torch.ones(per_rank * world, dtype=dtype, device=device)
        dst = torch.empty(per_rank * world, dtype=dtype, device=device)
        return (lambda: dist.all_to_all_single(dst, src)), per_rank * world * width
    if op == "broadcast":
        buf = torch.ones(elements, dtype=dtype, device=device)
        return (lambda: dist.broadcast(buf, src=0)), elements * width
    return None


def time_reps(fn, reps: int, warmup: int) -> list[float]:
    """Per-call seconds, one sample per rep, each rep entered from a barrier.

    The barrier is what makes the sample a measurement of the collective rather
    than of rank skew; it is also what adds launch latency, which is why small
    sizes are additionally burst-timed.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return samples


def time_bursts(fn, bursts: int, k: int) -> list[float]:
    """Mean per-call seconds within each burst: one barrier, then K calls back to
    back. No per-call barrier, so this is the steady-state cost a decode loop pays."""
    samples = []
    for _ in range(bursts):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / k)
    return samples


def library_paths() -> dict:
    """Resolve what is actually loaded, not what the module system promised.

    The net plugin is the component most likely to change underneath this study,
    so it is recorded by realpath rather than by name.
    """
    wanted = ("librccl", "librccl-net", "libfabric", "libamdhip", "libtorch_hip")
    found: dict[str, str] = {}
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            for line in handle:
                path = line.rstrip().split(" ")[-1]
                if not path.startswith("/"):
                    continue
                base = os.path.basename(path)
                for name in wanted:
                    if base.startswith(name) and name not in found:
                        found[name] = os.path.realpath(path)
    except OSError:
        pass
    return found


def effect_evidence(text: str) -> dict:
    """Pull the knob-took-effect facts out of an NCCL_DEBUG=INFO capture.

    Stage 1 runs with debug on and parses this; measurement slots run with
    NCCL_DEBUG=WARN and leave it empty, because per-rank INFO logging measurably
    changes the timings it is meant to explain.
    """
    if not text:
        return {}
    patterns = {
        "nchannels": r"(\d+)\s+coll channels",
        "buffsize": r"Setting buffsize to (\d+)",
        "net_plugin": r"Using network (\S+)",
        "msccl_algos": r"MSCCL.*?(\d+)\s+algorithm",
        "proto": r"Protocol\s+(\w+)",
        "algo": r"Algorithm\s+(\w+)",
    }
    out = {}
    for key, pattern in patterns.items():
        hits = re.findall(pattern, text)
        if hits:
            out[key] = sorted(set(hits))
    return out


def monitor_evidence() -> dict:
    """Which MR-cache monitor libfabric actually chose.

    Behavioural rather than log-based, reusing the trick that settled the monitor
    question in the bug study: userfaultfd shows as an anon_inode fd, kdreg2 as an
    open handle on its device node.
    """
    evidence = {"userfaultfd_fds": 0, "kdreg2_open": False}
    fd_dir = "/proc/self/fd"
    try:
        for name in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, name))
            except OSError:
                continue
            if "userfaultfd" in target:
                evidence["userfaultfd_fds"] += 1
            elif "kdreg2" in target:
                evidence["kdreg2_open"] = True
    except OSError:
        pass
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,all_to_all,broadcast")
    parser.add_argument("--min-bytes", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--serving-sizes", default="458752,917504,46137344",
                        help="exact EP all-to-all, TP all-reduce and prefill sizes")
    parser.add_argument("--fp32-sizes", default="1048576,134217728,1073741824",
                        help="kept for comparability with job 21791400")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--large-reps", type=int, default=8,
                        help="reps above 64 MiB, where each call costs tens of ms")
    parser.add_argument("--bursts", type=int, default=5)
    parser.add_argument("--burst-k", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the grid and exit; no GPU, no process group")
    args = parser.parse_args()

    extra = [int(s) for s in args.serving_sizes.split(",") if s.strip()]
    fp32_sizes = {int(s) for s in args.fp32_sizes.split(",") if s.strip()}
    sizes = size_grid(args.min_bytes, args.max_bytes, extra)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    if args.dry_run:
        print(json.dumps({"sizes": sizes, "n_sizes": len(sizes), "ops": ops,
                          "serving_sizes": extra, "fp32_sizes": sorted(fp32_sizes),
                          "latency_max": LATENCY_MAX, "a2a_max": A2A_MAX}, indent=2))
        return 0

    global torch, dist
    import torch as torch_mod
    import torch.distributed as dist_mod
    torch, dist = torch_mod, dist_mod

    rank = env_int("RANK", "SLURM_PROCID")
    world = env_int("WORLD_SIZE", "SLURM_NPROCS")
    local_rank = env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    device_index = 0 if torch.cuda.device_count() <= 1 else local_rank
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    dist.init_process_group(backend="nccl")

    rows = []
    for nbytes in sizes:
        for op in ops:
            dtypes = [("bf16", torch.bfloat16)]
            if nbytes in fp32_sizes:
                dtypes.append(("fp32", torch.float32))
            for dtype_name, dtype in dtypes:
                built = make_op(op, nbytes, dtype, device, world)
                if built is None:
                    rows.append({"op": op, "bytes": nbytes, "dtype": dtype_name,
                                 "skipped": "above a2a cap" if op == "all_to_all" else "unsupported"})
                    continue
                fn, moved = built
                row = {"op": op, "bytes": moved, "nominal_bytes": nbytes, "dtype": dtype_name}
                try:
                    reps = args.large_reps if nbytes > 64 * 1024**2 else args.reps
                    row["reps"] = reps
                    row["warmup"] = args.warmup
                    row["samples_seconds"] = time_reps(fn, reps, args.warmup)
                    if nbytes <= LATENCY_MAX:
                        row["burst_k"] = args.burst_k
                        row["burst_samples_seconds"] = time_bursts(fn, args.bursts, args.burst_k)
                except torch.cuda.OutOfMemoryError:
                    # A missing point is data. Recording the ceiling stops the
                    # summariser from quietly averaging a shorter curve.
                    row["skipped"] = "OOM"
                finally:
                    del built, fn
                    torch.cuda.empty_cache()
                rows.append(row)

    report = {
        "schema_version": 2,
        "variant": args.variant,
        "rank": rank,
        "world_size": world,
        "host": socket.gethostname(),
        "local_rank": local_rank,
        # Stamped by run_block.sh from the design; carried through so a row can be
        # traced back to the exact block, seed and permutation that produced it.
        "block": {k: os.environ.get(f"BLOCK_{k.upper()}")
                  for k in ("stage", "replicate", "id", "seed", "position", "role", "permutation")},
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "nodelist_raw": os.environ.get("SLURM_JOB_NODELIST"),
        "started_at_utc": os.environ.get("SLOT_STARTED_AT"),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # What run_block.sh meant to set, against what actually reached the rank.
        # singularity strips most of the environment unless it is passed through,
        # and a knob that never arrived is the most embarrassing possible null.
        "env_requested": json.loads(os.environ.get("SLOT_ENV_JSON", "{}")),
        "env_observed": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith(("NCCL_", "RCCL_", "FI_", "HSA_", "MIOPEN_"))},
        "effect_evidence": {**monitor_evidence(),
                            **effect_evidence(os.environ.get("SLOT_DEBUG_CAPTURE", ""))},
        "versions": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "device": torch.cuda.get_device_name(device_index),
            "container": os.environ.get("CONTAINER"),
            "libraries": library_paths(),
        },
        "rows": rows,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    # Position is part of the filename because the sentinel runs twice in every
    # block, first slot and last. Without it the closing sentinel overwrites the
    # opening one and the drift gate silently has nothing to compare.
    position = (os.environ.get("BLOCK_POSITION") or "0").zfill(2)
    out_path = os.path.join(
        args.results_dir, f"{args.variant}_p{position}_rank{rank:04d}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True)

    if rank == 0:
        measured = sum(1 for r in rows if "samples_seconds" in r)
        print(f"[{args.variant}] world={world} rows={len(rows)} measured={measured} -> {out_path}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
