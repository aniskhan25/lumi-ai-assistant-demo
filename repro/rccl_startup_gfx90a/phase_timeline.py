#!/usr/bin/env python3
"""Decompose vLLM multi-node startup into phases, per rank, from the server logs.

The report this case investigates says the stall is not tied to one phase. A single
elapsed number cannot test that claim, so this turns each rank's log into a timeline
anchored on markers vLLM emits itself -- including the exact line the reporter was
frozen on, `Using ['PYNCCL'] all-reduce`.

Markers are matched loosely and every one that is missing is reported as missing rather
than silently skipped: vLLM's log strings move between versions, and a marker that
stopped matching would otherwise look like a phase that took no time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime

# vLLM's own log prefix, e.g. "INFO 09-07 12:34:56 [core.py:123]".
# The year is absent, so timestamps are only ever used for differences.
LOG_TS = re.compile(r"^\w+\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")

# (phase name, regex, what a long gap before it would mean)
MARKERS = [
    ("launcher_start", r"=== launcher rank", "container up, before vLLM ran at all"),
    ("vllm_invoked", r"^Starting: vllm serve", "our own rank script handing off to vLLM"),
    ("engine_init", r"Initializing a[n]? V\d+ LLM engine|Initializing an LLM engine",
     "engine construction, config resolution"),
    ("dist_init", r"Initializing distributed environment|init_process_group|"
                  r"World size|distributed_init_method",
     "torch.distributed rendezvous over MASTER_ADDR:MASTER_PORT"),
    ("pynccl_allreduce", r"Using \[.*PYNCCL.*\] all-reduce|cuda_communicator",
     "RCCL communicator setup -- the reporter's 8-node freeze point"),
    ("weights_start", r"Loading safetensors|Loading weights|Starting to load model|"
                      r"0% Completed",
     "weight loader starting -- the reporter's 2-node freeze point"),
    ("weights_done", r"Model loading took|Loading model weights took|100% Completed",
     "the weight read itself (Lustre bandwidth lands here)"),
    ("kv_cache", r"Available KV cache memory|GPU KV cache size|Memory profiling",
     "memory profiling and KV cache sizing"),
    ("graph_capture", r"Capturing CUDA graph|Capturing cudagraphs|torch.compile|"
                      r"Compiling a graph",
     "graph capture / compile (MIOpen kernel cache misses land here)"),
    ("api_ready", r"Application startup complete|Starting vLLM API server|Uvicorn running",
     "API server accepting connections"),
]


def parse_ts(line: str) -> datetime | None:
    match = LOG_TS.match(line)
    if not match:
        return None
    try:
        # Year is arbitrary and identical everywhere; only differences are used.
        return datetime.strptime(f"2000-{match.group(1)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def timeline_for(path: str) -> dict:
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    hits: dict[str, dict] = {}

    with open(path, encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            ts = parse_ts(line)
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            for name, pattern, _ in MARKERS:
                if name in hits:
                    continue
                if re.search(pattern, line):
                    hits[name] = {
                        "line": lineno,
                        "at_seconds": (ts - first_ts).total_seconds()
                        if ts is not None and first_ts is not None else None,
                        "text": line.strip()[:200],
                    }

    ordered = [name for name, _, _ in MARKERS]
    reached = [n for n in ordered if n in hits]
    missing = [n for n in ordered if n not in hits]

    # Gaps between consecutive markers that were actually reached. The largest gap is
    # where this rank spent its startup, which is the whole point.
    gaps = []
    for earlier, later in zip(reached, reached[1:]):
        a, b = hits[earlier]["at_seconds"], hits[later]["at_seconds"]
        if a is not None and b is not None:
            gaps.append({"from": earlier, "to": later, "seconds": round(b - a, 1)})

    return {
        "log": os.path.basename(path),
        "total_log_seconds": round((last_ts - first_ts).total_seconds(), 1)
        if first_ts and last_ts else None,
        "markers": hits,
        "reached": reached,
        "missing": missing,
        "gaps": gaps,
        "largest_gap": max(gaps, key=lambda g: g["seconds"]) if gaps else None,
        # A rank that never reached api_ready is where the job actually hung.
        "stalled_after": reached[-1] if reached and "api_ready" not in hits else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True,
                        help="glob for rank logs, e.g. '/runtime/vllm_server_rank*.log'")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument("--startup-seconds", type=float, default=None,
                        help="wall-clock time to /v1/models, measured by the launcher")
    parser.add_argument("--verdict", default=None, help="READY or STALL, from the launcher")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.logs))
    if not paths:
        print(f"no logs matched {args.logs}")
        return 2

    report = {
        "variant": args.variant,
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES", "?"),
        "startup_seconds": args.startup_seconds,
        "verdict": args.verdict,
        "ranks": {},
    }
    for path in paths:
        rank = re.search(r"rank(\d+)", os.path.basename(path))
        report["ranks"][rank.group(1) if rank else os.path.basename(path)] = timeline_for(path)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"timeline_{args.variant}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(f"\n=== phase timeline: {args.variant} "
          f"(verdict={args.verdict}, startup={args.startup_seconds}s) ===")
    for rank, data in sorted(report["ranks"].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        print(f"\nrank {rank}  ({data['log']}, log spans {data['total_log_seconds']}s)")
        for name, _, meaning in MARKERS:
            hit = data["markers"].get(name)
            if hit is None:
                when = "NOT REACHED"
            elif hit["at_seconds"] is None:
                # Our own launcher lines carry no vLLM timestamp; they precede the
                # first timestamped line, so an offset would be invented, not measured.
                when = f"line {hit['line']}"
            else:
                when = f"{hit['at_seconds']:.0f}s"
            print(f"  {name:20s} {when:>12s}   {meaning}")
        if data["largest_gap"]:
            gap = data["largest_gap"]
            print(f"  -> longest phase: {gap['from']} -> {gap['to']} = {gap['seconds']}s")
        if data["stalled_after"]:
            print(f"  -> never became ready; last marker reached was {data['stalled_after']}")
        if data["missing"]:
            print(f"  -> markers never seen (may be a renamed log string): "
                  f"{', '.join(data['missing'])}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
