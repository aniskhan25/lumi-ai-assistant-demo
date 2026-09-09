#!/usr/bin/env python3
"""Turn the per-rank probe JSON into results/sweep.md.

Aggregation is by slowest rank, not by mean: a collective is only as fast as its
slowest participant, and one rank bootstrapping over the wrong interface is enough to
hold up all 64. Averaging would hide exactly the failure this case is chasing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import collections
from collections import defaultdict

# What a healthy variant lets us cross off. Kept next to the summariser so a reader of
# sweep.md never has to go find the hypothesis ledger to interpret a row.
WHAT_IT_RULES_OUT = {
    "baseline": "nothing -- this is the reference the report describes",
    "socket_ifname": "H-A: if this alone is healthy, RCCL was bootstrapping over a non-HSN interface",
    "gdr_level": "#4: GDR level was degrading the chosen path",
    "guide_pair": "H-A + #4 together, as the guide's lesson 5 sets them",
    "runtime_connect_off": "H-B: lazy per-communicator connection setup",
    "nchannels_8": "the reported workaround; healthy here only confirms the report",
    "nchannels_4": "#3: whether 8 is a threshold or just 'fewer'",
    "nchannels_16": "#3: the upper end of the dose-response",
    "nchannels_per_peer": "whether a per-peer cap suffices instead of a global one",
    "cxi_cq_and_sw_match": "#5: CQ depth and the hardware match cache",
    "proto_simple": "LL/LL128 protocol buffer setup",
    "cpu_bind": "#6: missing GCD-to-NUMA-to-NIC affinity",
    "container_older": "#10: container/plugin version (expected to change nothing)",
    "net_socket": "positive control -- must be healthy, or the harness is wrong",
    "gdr_cap": "whether the reported NCCL_MAX_NCHANNELS=8 workaround rescues the GDR hang",
    "gdr_runtime_connect": "whether eager connect rescues the GDR hang",
    "gdr_ifname": "whether pinning the interface rescues the GDR hang",
    "gdr_socket_net": "whether the hang is specific to the OFI/CXI path",
    "cxi_no_host_register": "FI_CXI_DISABLE_HOST_REGISTER=1, suggested in laifs-container-recipes#30",
    "mr_cache_monitor": "FI_MR_CACHE_MONITOR=userfaultfd, suggested in laifs-container-recipes#30",
    "cxi_both": "both recipes#30 mitigations together",
    "hpe_full": "HPE's full recommended RCCL set, via LUMI support",
    "hpe_minus_monitor": "HPE's set WITHOUT FI_MR_CACHE_MONITOR -- isolates whether the monitor is the crucial one",
    "mon_kdreg2": "explicit kdreg2 -- if this matches baseline, kdreg2 is the default and the culprit",
    "mon_memhooks": "explicit memhooks -- if this matches baseline, memhooks is the default and the culprit",
    "mon_disabled": "MR caching off entirely -- upper bound on what the cache costs, and a check that the cache is the mechanism",
}

# Ordered so the reader meets the phases in the order the process lives them.
PHASE_COLUMNS = [
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

# world_second is the warm control and is excluded: it is the denominator, not a cost.
STARTUP_PHASES = [p for p in PHASE_COLUMNS if p != "world_second"]

SLOW_FACTOR = 3.0

# Repeated attempts are recorded as "<variant>-r<n>" so nothing is overwritten.
REPEAT_SUFFIX = re.compile(r"-r\d+$")


def base_variant(name: str) -> str:
    return REPEAT_SUFFIX.sub("", name)


def load(results_dir: str) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    pattern = os.path.join(results_dir, "probe_*_rank*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {path}: {exc}")
            continue
        variant = record.get("variant") or re.sub(r"^probe_|_rank\d+\.json$", "", os.path.basename(path))
        by_variant[variant].append(record)
    return by_variant


def aggregate(records: list[dict]) -> dict:
    """Collapse one variant's ranks into the slowest-rank view."""
    phases: dict[str, float] = {}
    for name in PHASE_COLUMNS:
        seen = [r["phases"][name]["seconds"] for r in records if name in r.get("phases", {})]
        if seen:
            phases[name] = max(seen)

    stalled = [r for r in records if r.get("verdict") == "STALL"]
    errored = [r for r in records if r.get("verdict") == "INCOMPLETE"]
    startup = sum(phases.get(p, 0.0) for p in STARTUP_PHASES)

    return {
        "ranks": len(records),
        "world_size": records[0].get("world_size"),
        "nodes": records[0].get("nodes"),
        "job_id": records[0].get("job_id"),
        "phases": phases,
        "startup_seconds": round(startup, 3),
        "stalled_phases": sorted({r["stalled_phase"] for r in stalled if r.get("stalled_phase")}),
        "stalled_ranks": len(stalled),
        "incomplete_ranks": len(errored),
        "env": records[0].get("env", {}),
    }


def verdict(row: dict, reference: float | None) -> str:
    if row["stalled_ranks"]:
        return "STALL"
    if row["incomplete_ranks"]:
        return "ERROR"
    if reference is None or reference <= 0:
        return "FAST"
    return "SLOW" if row["startup_seconds"] > SLOW_FACTOR * reference else "FAST"


def render(rows: dict[str, dict], reference: float | None) -> str:
    out: list[str] = []
    any_row = next(iter(rows.values()))
    out.append("# RCCL startup sweep\n")
    out.append(
        f"Job `{any_row['job_id']}`, {any_row['nodes']} node(s), world size "
        f"{any_row['world_size']}. Times are seconds, taken from the **slowest rank** "
        f"in each phase.\n"
    )
    out.append(
        f"`startup_seconds` sums every phase except `world_second`, which is the warm "
        f"control used as the denominator. A variant is `SLOW` above "
        f"{SLOW_FACTOR:g}x the fastest variant here"
        + (f" ({reference:.2f}s).\n" if reference else ".\n")
    )

    header = ["variant", "verdict", "startup_s"] + PHASE_COLUMNS
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")

    # Healthy rows first: those are the candidate fixes, and that is what a reader wants.
    order = {"FAST": 0, "SLOW": 1, "STALL": 2, "ERROR": 3}
    for name, row in sorted(rows.items(), key=lambda kv: (order.get(kv[1]["verdict"], 9),
                                                          kv[1]["startup_seconds"])):
        cells = [f"`{name}`", f"**{row['verdict']}**", f"{row['startup_seconds']:.2f}"]
        for phase in PHASE_COLUMNS:
            value = row["phases"].get(phase)
            cells.append(f"{value:.2f}" if value is not None else "-")
        out.append("| " + " | ".join(cells) + " |")

    out.append("\n## What each variant rules out\n")
    out.append("| variant | verdict | rules out if healthy |")
    out.append("| --- | --- | --- |")
    for name, row in rows.items():
        rules = WHAT_IT_RULES_OUT.get(base_variant(name), "(undocumented)")
        out.append(f"| `{name}` | {row['verdict']} | {rules} |")

    stalls = {n: r for n, r in rows.items() if r["stalled_phases"]}
    if stalls:
        out.append("\n## Where the stalls landed\n")
        out.append(
            "A stall that lands in `init` points at H-A (bootstrap interface); one that "
            "lands in a `*_first` phase points at H-B (per-communicator connection "
            "setup).\n"
        )
        out.append("| variant | stalled phases | stalled ranks |")
        out.append("| --- | --- | --- |")
        for name, row in stalls.items():
            out.append(f"| `{name}` | {', '.join(row['stalled_phases'])} | "
                       f"{row['stalled_ranks']}/{row['ranks']} |")

    repeats = collections.defaultdict(list)
    for name, row in rows.items():
        repeats[base_variant(name)].append(row)
    if any(len(v) > 1 for v in repeats.values()):
        out.append("\n## Stall rate across repeats\n")
        out.append(
            "A variant that stalls on some attempts and not others is a race, not an "
            "effect of its setting. That distinction is the whole reason for repeating.\n"
        )
        out.append("| variant | attempts | stalled attempts | phases seen stalling |")
        out.append("| --- | --- | --- | --- |")
        for name, group in sorted(repeats.items()):
            stalled = [r for r in group if r["stalled_ranks"]]
            phases = sorted({p for r in group for p in r["stalled_phases"]})
            out.append(f"| `{name}` | {len(group)} | {len(stalled)} | "
                       f"{', '.join(phases) if phases else '-'} |")

    out.append("\n## Environment per variant\n")
    tracked = ("NCCL_SOCKET_IFNAME", "NCCL_NET_GDR_LEVEL", "NCCL_RUNTIME_CONNECT",
               "NCCL_MAX_NCHANNELS", "NCCL_NCHANNELS_PER_NET_PEER", "NCCL_PROTO",
               "NCCL_NET", "FI_CXI_DEFAULT_CQ_SIZE", "FI_CXI_RX_MATCH_MODE")
    out.append("| variant | " + " | ".join(k.replace("NCCL_", "").replace("FI_CXI_", "CXI_")
                                           for k in tracked) + " |")
    out.append("| " + " | ".join(["---"] * (len(tracked) + 1)) + " |")
    for name, row in rows.items():
        cells = [f"`{name}`"] + [row["env"].get(k, "-") for k in tracked]
        out.append("| " + " | ".join(cells) + " |")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = parser.parse_args()

    by_variant = load(args.results_dir)
    if not by_variant:
        print(f"no probe_*.json found in {args.results_dir}")
        return 2

    rows = {name: aggregate(records) for name, records in by_variant.items()}

    healthy = [r["startup_seconds"] for r in rows.values()
               if not r["stalled_ranks"] and not r["incomplete_ranks"]]
    reference = min(healthy) if healthy else None
    for row in rows.values():
        row["verdict"] = verdict(row, reference)

    out_path = os.path.join(args.results_dir, "sweep.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(render(rows, reference))

    print(f"wrote {out_path}\n")
    for name, row in sorted(rows.items(), key=lambda kv: kv[1]["startup_seconds"]):
        stalled = f"  stalled in {','.join(row['stalled_phases'])}" if row["stalled_phases"] else ""
        print(f"  {name:22s} {row['verdict']:6s} startup={row['startup_seconds']:8.2f}s{stalled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
