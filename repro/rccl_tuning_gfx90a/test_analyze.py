#!/usr/bin/env python3
"""End-to-end check of the analysis against synthetic data whose answer is known.

The point of the second suite is the one the plan calls the self-check: comparing
the reference against itself, across allocations, must come back with no winner. A
pipeline that reports a winner on null input would have reported one on the real
data too, and nobody would have been able to tell.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile

import analyze
import designs
import stats

OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast"]
SIZES = [8, 1024, 8192, 16384, 262144, 458752, 917504, 1048576,
         4194304, 46137344, 134217728, 268435456, 1073741824]


def synth_rows(rng: random.Random, speed: float, jitter: float) -> list[dict]:
    """A plausible time model: fixed latency plus size over bandwidth.

    `speed` > 1 means this variant is faster. Everything it touches scales, so a
    band gain and the promotion scalar move together the way they would in reality.
    """
    rows = []
    for nbytes in SIZES:
        for op in OPS:
            if op == "all_to_all" and nbytes > 256 * 1024**2:
                rows.append({"op": op, "bytes": nbytes, "nominal_bytes": nbytes,
                             "dtype": "bf16", "skipped": "above a2a cap"})
                continue
            base = (30e-6 + nbytes / 45e9) / speed
            samples = [base * (1 + rng.gauss(0, jitter)) for _ in range(20)]
            row = {"op": op, "bytes": nbytes, "nominal_bytes": nbytes, "dtype": "bf16",
                   "samples_seconds": samples, "reps": 20, "warmup": 5}
            if nbytes <= 1024 * 1024:
                row["burst_k"] = 50
                row["burst_samples_seconds"] = [base * 0.85 * (1 + rng.gauss(0, jitter))
                                                for _ in range(5)]
            rows.append(row)
    return rows


def write_slot(root: str, job_id: str, slot: dict, speed: float, rng: random.Random,
               n_ranks: int = 4, jitter: float = 0.01, env_observed=None) -> None:
    job_dir = os.path.join(root, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)
    for rank in range(n_ranks):
        # Rank-to-rank skew, so the max-across-ranks aggregation has something to do.
        rank_speed = speed * (1 - 0.002 * rank)
        report = {
            "schema_version": 2, "variant": slot["name"], "rank": rank, "world_size": n_ranks,
            "host": f"nid00{1000 + rank}", "local_rank": rank,
            "block": {"stage": slot.get("stage", "2"), "replicate": "1", "id": "1",
                      "seed": "0", "position": str(slot["position"]), "role": slot["role"],
                      "permutation": "synthetic"},
            "job_id": job_id, "partition": "dev-g", "nodes": "4",
            "nodelist_raw": "nid[001000-001003]",
            "env_requested": slot["env"],
            "env_observed": slot["env"] if env_observed is None else env_observed,
            "effect_evidence": {"userfaultfd_fds": 1, "kdreg2_open": False},
            "versions": {"torch": "2.10.0", "hip": "7.0"},
            "rows": synth_rows(rng, rank_speed, jitter),
        }
        name = f"{slot['name']}_p{str(slot['position']).zfill(2)}_rank{rank:04d}.json"
        with open(os.path.join(job_dir, name), "w") as handle:
            json.dump(report, handle)


def run_analysis(root: str, extra: list[str] | None = None) -> str:
    dirs = sorted(d for d in os.listdir(root) if d.startswith("job_"))
    argv = sys.argv
    out = os.path.join(root, "tuning.md")
    sys.argv = ["analyze.py", "--out", out, "--cv-pos-pct", "2.0"] + [
        arg for d in dirs for arg in ("--results-dir", os.path.join(root, d))] + (extra or [])
    try:
        analyze.main()
    finally:
        sys.argv = argv
    return open(out, encoding="utf-8").read()


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  -- ' + detail if detail and not condition else ''}")
    return condition


def test_null_reports_no_winner() -> bool:
    """The self-check. Every slot is the reference; nothing may be promoted."""
    root = tempfile.mkdtemp()
    try:
        rng = random.Random(1)
        for job in range(1, 5):
            for pos in range(1, 7):
                role = "sentinel" if pos in (1, 6) else "design"
                name = "ref" if role == "sentinel" else f"null{pos}"
                # Allocation-level offset: the term pairing is supposed to cancel.
                offset = 1.0 + rng.gauss(0, 0.06)
                write_slot(root, f"9{job:03d}", {"name": name, "role": role, "position": pos,
                                                 "env": dict(designs.REF_ENV), "stage": "2"},
                           offset, rng)
        report = run_analysis(root)
        ok = check("null input promotes nothing", "| promote |" not in report)
        ok &= check("null input kills nothing", "| kill |" not in report)
        ok &= check("controls section is present", "## Controls" in report)
        ok &= check("environment was verified for every slot",
                    "environment did not reach the ranks" not in report)
        return ok
    finally:
        shutil.rmtree(root)


def test_injected_effect_is_found_and_sized() -> bool:
    root = tempfile.mkdtemp()
    try:
        rng = random.Random(2)
        for job in range(1, 5):
            offset = 1.0 + rng.gauss(0, 0.06)
            for pos, (name, speed) in enumerate(
                    [("ref", 1.0), ("fast8", 1.08), ("slow6", 0.94), ("noeffect", 1.0),
                     ("ref", 1.0)], start=1):
                role = "sentinel" if name == "ref" else "design"
                write_slot(root, f"8{job:03d}", {"name": name, "role": role, "position": pos,
                                                 "env": dict(designs.REF_ENV), "stage": "2"},
                           offset * speed, rng)
        report = run_analysis(root)

        line = next((l for l in report.splitlines() if l.startswith("| `fast8` |")), "")
        ok = check("the +8% variant is measured near +8%",
                   "+7." in line or "+8." in line, line.strip())
        ok &= check("and it is promoted", "promote" in line, line.strip())

        slow = next((l for l in report.splitlines() if l.startswith("| `slow6` |")), "")
        ok &= check("the -6% variant is measured negative", "-" in slow.split("|")[3], slow.strip())
        ok &= check("and it is killed", "kill" in slow, slow.strip())

        flat = next((l for l in report.splitlines() if l.startswith("| `noeffect` |")), "")
        ok &= check("the null variant is not promoted", "promote" not in flat, flat.strip())
        return ok
    finally:
        shutil.rmtree(root)


def test_env_mismatch_blocks_a_verdict() -> bool:
    """A knob that never reached the ranks must produce UNVERIFIED, not a number."""
    root = tempfile.mkdtemp()
    try:
        rng = random.Random(3)
        env = dict(designs.REF_ENV, NCCL_BUFFSIZE="8388608")
        for pos, (name, role) in enumerate(
                [("ref", "sentinel"), ("ghost", "design"), ("ref", "sentinel")], start=1):
            write_slot(root, "7001", {"name": name, "role": role, "position": pos,
                                      "env": env if name == "ghost" else dict(designs.REF_ENV),
                                      "stage": "2"},
                       1.0, rng,
                       # The rank saw the reference environment: the knob never arrived.
                       env_observed=dict(designs.REF_ENV) if name == "ghost" else None)
        report = run_analysis(root)
        ok = check("the lost knob is reported as unverified",
                   "environment did not reach the ranks" in report)
        ok &= check("and it is named", "`ghost`" in report and "NCCL_BUFFSIZE" in report)
        ok &= check("and it is excluded from the gains table",
                    not any(l.startswith("| `ghost` |") for l in report.splitlines()))
        return ok
    finally:
        shutil.rmtree(root)


def test_controls_fire() -> bool:
    root = tempfile.mkdtemp()
    try:
        rng = random.Random(4)
        # cap4 at half speed: the known-sign control behaving as it must.
        for pos, (name, role, speed) in enumerate(
                [("ref", "sentinel", 1.0), ("cap4", "control", 0.45),
                 ("sham_b1", "sham", 1.0), ("ref", "sentinel", 1.0)], start=1):
            write_slot(root, "6001", {"name": name, "role": role, "position": pos,
                                      "env": dict(designs.REF_ENV), "stage": "2"},
                       speed, rng)
        report = run_analysis(root)
        ok = check("the known-sign control is reported and passes",
                   "known-sign `cap4`" in report and "HARNESS NOT APPLYING ENV" not in report)
        ok &= check("the sham is reported", "sham (`sham_b1`)" in report)
        ok &= check("sentinel drift is reported against a bound", "sentinel drift" in report)
        return ok
    finally:
        shutil.rmtree(root)


def test_noise_stage_reports_mde() -> bool:
    root = tempfile.mkdtemp()
    try:
        rng = random.Random(6)
        for job in range(1, 6):
            offset = 1.0 + rng.gauss(0, 0.05)
            for pos in range(1, 9):
                # Slot 1 pays a cold-start step the later slots do not.
                step = 0.93 if pos == 1 else 1.0
                write_slot(root, f"5{job:03d}",
                           {"name": "ref", "role": "sentinel", "position": pos,
                            "env": dict(designs.REF_ENV), "stage": "0"},
                           offset * step, rng)
        report = run_analysis(root)
        ok = check("stage 0 section is produced", "## Stage 0 — noise floor" in report)
        ok &= check("both variance components are reported",
                    "| allocation |" in report and "| position |" in report)
        ok &= check("the MDE ladder is produced", "allocations per arm" in report)
        ok &= check("the slot-1 step is separated from the slope", "Slot-1 step" in report)
        return ok
    finally:
        shutil.rmtree(root)


def main() -> int:
    suites = [("a null comparison produces no winner", test_null_reports_no_winner),
              ("a known effect is found and correctly sized", test_injected_effect_is_found_and_sized),
              ("a knob that never arrived blocks its verdict", test_env_mismatch_blocks_a_verdict),
              ("controls fire", test_controls_fire),
              ("stage 0 reports variance and the MDE", test_noise_stage_reports_mde)]
    ok = True
    for label, fn in suites:
        print(f"\n{label}")
        ok &= fn()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
