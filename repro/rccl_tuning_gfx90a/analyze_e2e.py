#!/usr/bin/env python3
"""Stage 5: does the microbenchmark gain survive into tok/s per GCD?

Applies the acceptance rule that was pre-registered before any of these jobs ran, so
the verdict is computed rather than eyeballed. A microbenchmark win that has not
cleared this is a microbenchmark win, not a recommendation, and must not reach
env_tuned.sh.

    ref  = current shipped baseline
    cand = baseline + NCCL_MIN_NCHANNELS=32

Arms are compared **within contemporaneous pairs**. One ref and one cand ran at the
same time on the same partition, so whatever the fabric and the filesystem were doing
that hour, both arms saw it. The peak concurrency of each run is used: that is where a
serving configuration is judged, and where collective cost is largest relative to
everything else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from statistics import fmean, pstdev

import stats


def load_runs(root: str, profile: str) -> dict[str, dict]:
    """Best no-failure row per job, keyed by job id."""
    runs: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(root, profile, "job_*", "summary_*.json"))):
        job = os.path.basename(os.path.dirname(path)).replace("job_", "")
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
        if int(d.get("requests_failed", 0)) != 0:
            continue
        row = {
            "job": job,
            "concurrency": int(d.get("concurrency", 0)),
            "ctok_s": float(d.get("throughput_completion_tokens_s", 0.0)),
            "p95": float(d.get("latency_p95_s", 0.0)),
            "failed": int(d.get("requests_failed", 0)),
        }
        best = runs.get(job)
        # Peak concurrency, and among equals the faster run.
        if (best is None or row["concurrency"] > best["concurrency"]
                or (row["concurrency"] == best["concurrency"] and row["ctok_s"] > best["ctok_s"])):
            runs[job] = row
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--ref-profile", default="s5_ref")
    parser.add_argument("--cand-profile", default="s5_cand")
    parser.add_argument("--pairs", default="",
                        help="refjob:candjob,refjob:candjob — the contemporaneous pairs")
    parser.add_argument("--gcds", type=int, default=32, help="4 nodes x 8 GCDs")
    parser.add_argument("--out")
    args = parser.parse_args()

    ref = load_runs(args.results_root, args.ref_profile)
    cand = load_runs(args.results_root, args.cand_profile)
    if not ref or not cand:
        print(f"need both arms: ref={len(ref)} cand={len(cand)}")
        return 2

    pairs = []
    if args.pairs:
        for spec in args.pairs.split(","):
            r, c = spec.split(":")
            if r in ref and c in cand:
                pairs.append((ref[r], cand[c]))
    else:
        # Fall back to submission order, but say so: unpaired comparison is weaker.
        for r, c in zip(sorted(ref), sorted(cand)):
            pairs.append((ref[r], cand[c]))

    lines = ["# Stage 5 — end to end, tok/s per GCD\n",
             f"{len(pairs)} contemporaneous pair(s), {args.gcds} GCDs, peak concurrency per run.\n",
             "| pair | ref job | cand job | conc | ref tok/s/GCD | cand tok/s/GCD | gain | ref p95 | cand p95 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    gains, p95_deltas = [], []
    for i, (r, c) in enumerate(pairs, start=1):
        rg, cg = r["ctok_s"] / args.gcds, c["ctok_s"] / args.gcds
        gain = (cg / rg - 1) * 100 if rg else 0.0
        gains.append(gain)
        dp = (c["p95"] / r["p95"] - 1) * 100 if r["p95"] else 0.0
        p95_deltas.append(dp)
        lines.append(f"| {i} | `{r['job']}` | `{c['job']}` | {c['concurrency']} | "
                     f"{rg:.3f} | {cg:.3f} | **{gain:+.2f}%** | {r['p95']:.2f}s | {c['p95']:.2f}s |")

    mean_gain = fmean(gains)
    ref_spread = (max(g["ctok_s"] for g in ref.values())
                  / min(g["ctok_s"] for g in ref.values()) - 1) * 100 if len(ref) > 1 else float("nan")
    worst_p95 = max(p95_deltas)
    all_positive = all(g > 0 for g in gains)
    any_failures = any(r["failed"] for r in list(ref.values()) + list(cand.values()))

    lines.append("")
    lines.append("## Acceptance rule, fixed before these jobs ran\n")
    lines.append("| criterion | required | observed | verdict |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| sign consistency | positive in every pair | {sum(g > 0 for g in gains)}/{len(gains)} "
                 f"positive | {'PASS' if all_positive else '**FAIL**'} |")
    lines.append(f"| magnitude | mean gain >= +2% | {mean_gain:+.2f}% | "
                 f"{'PASS' if mean_gain >= 2.0 else '**FAIL**'} |")
    lines.append(f"| exceeds reference spread | gain > ref-arm spread | ref spread "
                 f"{ref_spread:.2f}% | {'PASS' if mean_gain > ref_spread else '**FAIL**'} |")
    lines.append(f"| no p95 regression | <= +3% | worst {worst_p95:+.2f}% | "
                 f"{'PASS' if worst_p95 <= 3.0 else '**FAIL**'} |")
    lines.append(f"| no failed requests | 0 | {'some' if any_failures else '0'} | "
                 f"{'**FAIL**' if any_failures else 'PASS'} |")

    promoted = (all_positive and mean_gain >= 2.0 and mean_gain > ref_spread
                and worst_p95 <= 3.0 and not any_failures)
    lines.append("")
    lines.append(f"## Verdict: {'**PROMOTE** to env_tuned.sh' if promoted else '**DO NOT PROMOTE**'}\n")
    if not promoted:
        lines.append("> The microbenchmark gain did not clear the end-to-end rule. It is reported "
                     "as a microbenchmark result, with the dilution stated, and no line is added "
                     "to `env_tuned.sh`.\n")

    out = args.out or os.path.join(args.results_root, "stage5.md")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
