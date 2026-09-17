#!/usr/bin/env python3
"""Off-cluster checks on the design. Run before submitting anything:

    python3 repro/rccl_tuning_gfx90a/test_designs.py

These are the properties the whole study leans on. A generator row transcribed
from a textbook is exactly the kind of thing that is wrong in a way no amount of
GPU time reveals, so it gets checked rather than trusted.
"""
from __future__ import annotations

import sys

import designs


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  -- ' + detail if detail and not condition else ''}")
    return condition


def test_pb20() -> bool:
    d = designs.pb20()
    n_rows, n_cols = len(d), len(d[0])
    ok = check("PB-20 is 20 runs x 19 columns", (n_rows, n_cols) == (20, 19),
               f"got {n_rows}x{n_cols}")
    ok &= check("every entry is +/-1", all(v in (1, -1) for r in d for v in r))

    counts = {sum(1 for r in d if r[j] == 1) for j in range(n_cols)}
    ok &= check("every column is balanced 10 high / 10 low", counts == {10}, f"got {counts}")

    bad = [(a, b) for a in range(n_cols) for b in range(a + 1, n_cols)
           if sum(d[i][a] * d[i][b] for i in range(n_rows)) != 0]
    ok &= check("all 171 column pairs orthogonal", not bad, f"{len(bad)} bad pairs")

    ok &= check("no two columns identical",
                len({tuple(r[j] for r in d) for j in range(n_cols)}) == n_cols)
    ok &= check("room for every factor plus a blocking dummy",
                len(designs.FACTORS) < n_cols, f"{len(designs.FACTORS)} factors")
    return ok


def test_blocking() -> bool:
    blocks = designs.screen_blocks(replicate=1, seed=7)
    ok = check("screen splits into 2 allocations", len(blocks) == 2)

    design_slots = [s for b in blocks for s in b["slots"] if s["role"] == "design"]
    ok &= check("all 20 PB runs appear exactly once",
                sorted(s["run_index"] for s in design_slots) == list(range(20)))

    for b in blocks:
        got = [s for s in b["slots"] if s["role"] == "design"]
        ok &= check(f"block {b['block']} holds 10 design runs", len(got) == 10, f"got {len(got)}")
        for factor in designs.FACTORS:
            highs = sum(1 for s in got if s["levels"][factor.key] == 1)
            ok &= check(f"block {b['block']}: {factor.key} balanced 5/5 within the allocation",
                        highs == 5, f"got {highs} high")
    return ok


def test_block_structure() -> bool:
    ok = True
    for b in designs.screen_blocks(replicate=2, seed=3):
        slots = b["slots"]
        ok &= check(f"block {b['block']} opens on a discarded warm-up slot",
                    slots[0]["role"] == "warmup")
        ok &= check(f"block {b['block']} brackets the body with warm sentinels",
                    slots[1]["role"] == "sentinel" and slots[-1]["role"] == "sentinel")
        ok &= check(f"block {b['block']} carries exactly one sham",
                    sum(1 for s in slots if s["role"] == "sham") == 1)
        ok &= check(f"block {b['block']} positions are 1..n in order",
                    [s["position"] for s in slots] == list(range(1, len(slots) + 1)))
        ok &= check(f"block {b['block']} sham env is identical to the sentinel",
                    next(s for s in slots if s["role"] == "sham")["env"]
                    == next(s for s in slots if s["role"] == "sentinel")["env"])
    return ok


def test_reproducible_and_varying() -> bool:
    order = lambda rep, seed: [s["name"] for s in designs.screen_blocks(rep, seed)[0]["slots"]]
    ok = check("same (replicate, seed) gives the same slot order", order(1, 42) == order(1, 42))
    ok &= check("different replicates give different orders", order(1, 42) != order(2, 42))
    ok &= check("different seeds give different orders", order(1, 42) != order(1, 43))

    grids = [designs.algo_proto_block(1, 0), designs.channels_buffsize_block(1, 0)]
    for g in grids:
        ok &= check(f"stage {g['stage']} grid is 9 cells plus the cap4 control",
                    sum(1 for s in g["slots"] if s["role"] in ("design", "control")) == 10)
        ok &= check(f"stage {g['stage']} carries the cap4 known-sign control",
                    any(s["name"] == "cap4" for s in g["slots"]))
    return ok


def test_env_hygiene() -> bool:
    slots = [s for b in designs.screen_blocks(1, 0) for s in b["slots"]]
    ok = check("every slot inherits the bug study's monitor fix",
               all(s["env"].get("FI_MR_CACHE_MONITOR") for s in slots))
    ok &= check("MANAGED_VARS covers every key any slot sets",
                all(k in designs.MANAGED_VARS for s in slots for k in s["env"]))
    ok &= check("NCCL_NET_GDR_LEVEL is never set anywhere",
                not any("NCCL_NET_GDR_LEVEL" in s["env"] for s in slots),
                "it hangs deterministically -- jobs 21790392, 21790393, 21794114")

    monitor = next(f for f in designs.FACTORS if f.key == "mr_monitor")
    ok &= check("the monitor factor never leaves the monitor unset",
                bool(monitor.low) and bool(monitor.high),
                "an unset level would reintroduce the memhooks hang")

    bind = next(f for f in designs.FACTORS if f.key == "cpu_bind")
    ok &= check("cpu_bind varies srun flags, not environment",
                bind.low_flags != bind.high_flags and not bind.low and not bind.high)
    return ok


def main() -> int:
    suites = [("PB-20 construction", test_pb20),
              ("blocking on a dummy column", test_blocking),
              ("block structure and controls", test_block_structure),
              ("reproducibility", test_reproducible_and_varying),
              ("environment hygiene", test_env_hygiene)]
    ok = True
    for label, fn in suites:
        print(f"\n{label}")
        ok &= fn()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
