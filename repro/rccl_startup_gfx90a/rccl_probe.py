#!/usr/bin/env python3
"""Time RCCL startup phase by phase, with no model and no vLLM in the picture.

The user report that this case investigates says the stall "isn't tied to one phase"
-- different runs freeze at different stages of vLLM startup. Two mechanisms explain
that, and this probe is built to tell them apart:

  H-A  RCCL's out-of-band bootstrap picks a non-HSN interface, because
       NCCL_SOCKET_IFNAME is unset. Cost lands in `init`.
  H-B  RCCL defers per-peer connection setup to the first collective that needs it
       (NCCL_RUNTIME_CONNECT), so every *new communicator* pays a fresh burst. Cost
       lands in `first`, and again in each `*_first` below.

So the phases are deliberately ordered to separate "one-time init" from "per-communicator
setup", and no barrier is inserted before `world_first` -- that all_reduce must be the
genuinely first collective or the measurement is worthless.

Each rank writes its own JSON. A rank that hangs is still a recorded result: a watchdog
thread writes the STALL verdict and calls os._exit, because a thread blocked inside RCCL
cannot be interrupted from Python and waiting it out would burn the whole allocation.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import socket
import sys
import threading
import time

import torch
import torch.distributed as dist


# Phases whose name ends in `_first` are the ones a fresh-communicator burst shows up
# in; `world_second` is the control that must stay flat if H-B holds.
PHASE_ORDER = [
    "init",
    "world_first",
    "world_second",
    "fresh_world_first_create",
    "fresh_world_first",
    "tp_like_create",
    "tp_like_first",
    "pp_like_create",
    "pp_like_first",
    "many_comms",
]

WHAT_IT_MEASURES = {
    "init": "bootstrap/rendezvous only -- H-A lands here",
    "world_first": "first collective anywhere: connection setup for the default comm",
    "world_second": "same comm, already warm -- control, must be fast",
    "fresh_world_first_create": "building a second comm object, before any traffic on it",
    "fresh_world_first": "a second comm over the same peers -- H-B lands here",
    "tp_like_create": "building the intra-node groups",
    "tp_like_first": "intra-node group, mirrors vLLM's TP comm",
    "pp_like_create": "building the cross-node groups",
    "pp_like_first": "cross-node group, mirrors vLLM's PP comm",
    "many_comms": "N more communicators, each used once -- tests whether cost grows "
                  "with communicator count, the one axis vLLM pushes much harder than "
                  "a microbenchmark does",
}


def env_int(*names: str, default: int | None = None) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return int(value)
    if default is None:
        raise RuntimeError(f"none of {names} set in the environment")
    return default


class PhaseTimer:
    """Times phases and guarantees a result on disk even when a phase never returns."""

    def __init__(self, out_path: str, stall_timeout_s: float, meta: dict):
        self.out_path = out_path
        self.stall_timeout_s = stall_timeout_s
        self.record = dict(meta)
        self.record["phases"] = {}
        self.record["verdict"] = "INCOMPLETE"
        self.record["stalled_phase"] = None
        self._done = threading.Event()
        self._current = None
        self._started_at = None

    def _watch(self) -> None:
        # Wait for the phase to finish; if it does not, this rank is the evidence.
        if self._done.wait(self.stall_timeout_s):
            return
        elapsed = time.perf_counter() - self._started_at
        self.record["phases"][self._current] = {
            "seconds": round(elapsed, 3),
            "measures": WHAT_IT_MEASURES.get(self._current, ""),
            "stalled": True,
        }
        self.record["verdict"] = "STALL"
        self.record["stalled_phase"] = self._current
        self.write()
        sys.stderr.write(
            f"STALL in phase {self._current} after {elapsed:.1f}s; "
            f"traceback follows, then aborting this rank\n"
        )
        sys.stderr.flush()
        faulthandler.dump_traceback(all_threads=True)
        # A rank stuck inside RCCL cannot be unwound, so leave rather than hold nodes.
        os._exit(75)

    def run(self, name: str, fn):
        self._current = name
        self._done.clear()
        self._started_at = time.perf_counter()
        watcher = threading.Thread(target=self._watch, name=f"watchdog-{name}", daemon=True)
        watcher.start()
        try:
            result = fn()
        finally:
            self._done.set()
        elapsed = time.perf_counter() - self._started_at
        self.record["phases"][name] = {
            "seconds": round(elapsed, 3),
            "measures": WHAT_IT_MEASURES.get(name, ""),
            "stalled": False,
        }
        return result

    def write(self) -> None:
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        tmp = f"{self.out_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.record, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.out_path)


def tracked_env() -> dict:
    """Only the variables this investigation moves, so a result names its own config."""
    prefixes = ("NCCL_", "RCCL_", "FI_", "OFI_", "CXI_", "HSA_", "ROCR_", "HIP_")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline", help="name of the env variant under test")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    parser.add_argument(
        "--stall-timeout-s",
        type=float,
        default=float(os.environ.get("STALL_TIMEOUT_S", "300")),
        help="a phase exceeding this is recorded as STALL and the rank aborts",
    )
    parser.add_argument("--many-comms", type=int,
                        default=int(os.environ.get("MANY_COMMS", "0")),
                        help="create this many extra communicators and use each once; "
                             "0 disables the phase")
    parser.add_argument("--tensor-mib", type=float, default=32.0,
                        help="all_reduce payload; big enough to need real channels, "
                             "small enough that time is setup, not bandwidth")
    args = parser.parse_args()

    rank = env_int("RANK", "SLURM_PROCID")
    world_size = env_int("WORLD_SIZE", "SLURM_NPROCS")
    local_rank = env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    local_world = env_int("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", default=8)

    # Slurm hands each task its own GCD, so the visible device is usually index 0.
    device_index = 0 if torch.cuda.device_count() <= 1 else local_rank
    torch.cuda.set_device(device_index)

    out_path = os.path.join(args.results_dir, f"probe_{args.variant}_rank{rank}.json")
    timer = PhaseTimer(
        out_path,
        args.stall_timeout_s,
        {
            "variant": args.variant,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "nodes": os.environ.get("SLURM_JOB_NUM_NODES", "?"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gcn_arch": torch.cuda.get_device_properties(device_index).gcnArchName,
            "stall_timeout_s": args.stall_timeout_s,
            "tensor_mib": args.tensor_mib,
            "many_comms": args.many_comms,
            "env": tracked_env(),
        },
    )
    # Any early crash should still leave something behind to read.
    timer.write()

    elements = max(1, int(args.tensor_mib * 1024 * 1024 / 4))
    payload = torch.ones(elements, dtype=torch.float32, device=f"cuda:{device_index}")

    def allreduce(group=None):
        dist.all_reduce(payload, group=group)
        torch.cuda.synchronize()

    timer.run("init", lambda: dist.init_process_group(backend="nccl"))

    # No barrier here on purpose: this must be the first collective of the process.
    timer.run("world_first", lambda: allreduce())
    timer.run("world_second", lambda: allreduce())

    # A second communicator over the *same* peers. vLLM does exactly this (TP, PP and
    # world groups overlap), so if this is expensive, "stalls at different phases" needs
    # no further explanation.
    fresh = timer.run("fresh_world_first_create",
                      lambda: dist.new_group(ranks=list(range(world_size))))
    timer.run("fresh_world_first", lambda: allreduce(fresh))

    # Mirror vLLM's real split: TP within a node, PP across nodes at the same local rank.
    # Group construction is itself timed, because dist.new_group is collective and can
    # stall; running it outside a timed phase would mean no watchdog and a job that
    # hangs to the wall clock.
    n_nodes = max(1, world_size // local_world)

    def build_tp():
        mine = None
        for node in range(n_nodes):
            ranks = list(range(node * local_world, (node + 1) * local_world))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                mine = group
        return mine

    def build_pp():
        mine = None
        for slot in range(local_world):
            ranks = [slot + node * local_world for node in range(n_nodes)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                mine = group
        return mine

    tp_group = timer.run("tp_like_create", build_tp)
    timer.run("tp_like_first", lambda: allreduce(tp_group))

    pp_group = timer.run("pp_like_create", build_pp)
    timer.run("pp_like_first", lambda: allreduce(pp_group))

    if args.many_comms > 0:
        def many():
            # Each communicator is used once, which is what vLLM's startup effectively
            # does across its TP/PP/world/all2all groups. Held in a list so none is
            # garbage-collected mid-phase, which would destroy the comm being measured.
            groups = []
            for _ in range(args.many_comms):
                group = dist.new_group(ranks=list(range(world_size)))
                groups.append(group)
                dist.all_reduce(payload, group=group)
            torch.cuda.synchronize()
            return len(groups)

        timer.run("many_comms", many)

    phases = timer.record["phases"]
    # The signature that matters is a ratio, not an absolute: a fresh communicator
    # costing about what the first one did is the H-B fingerprint.
    warm = phases["world_second"]["seconds"]
    timer.record["ratios"] = {
        "world_first_over_warm": round(phases["world_first"]["seconds"] / warm, 1) if warm else None,
        "fresh_world_over_warm": round(phases["fresh_world_first"]["seconds"] / warm, 1) if warm else None,
        "pp_like_over_warm": round(phases["pp_like_first"]["seconds"] / warm, 1) if warm else None,
    }
    timer.record["verdict"] = "COMPLETED"
    timer.write()

    if rank == 0:
        print(f"[{args.variant}] world_size={world_size} phases (s):")
        for name in PHASE_ORDER:
            if name in phases:
                print(f"  {name:24s} {phases[name]['seconds']:9.3f}   {WHAT_IT_MEASURES[name]}")
        print(f"  ratios vs warm all_reduce: {timer.record['ratios']}")

    # No closing barrier on purpose: if any rank aborted on a stall, a barrier here
    # would block every survivor with no watchdog left to rescue them.
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
