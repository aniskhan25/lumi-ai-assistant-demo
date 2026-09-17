#!/usr/bin/env python3
"""Turn a block's per-rank JSON into a verdict, or refuse to.

Descends from summarize_bandwidth.py and summarize_sweep.py. Three things the
ancestors could not do, each of which this study needs:

  * **Paired contrasts against an in-block sentinel.** Allocation-to-allocation
    spread on LUMI is the dominant noise term -- different Dragonfly spans,
    different neighbours. Comparing a variant to the sentinel that ran in its own
    allocation cancels it. Comparing raw numbers across allocations does not, and
    that is how the bug study produced five findings it had to retract.
  * **Bands aimed at this repo's traffic.** The ancestor headlines 128 MiB - 1 GiB.
    A vLLM decode step all-reduces ~896 KiB and all-to-alls ~448 KiB, so the old
    headline band is ~150x larger than anything this repo actually sends.
  * **Refusing to answer.** A knob that never reached the rank, or that libfabric
    silently ignored, produces a clean null that looks exactly like a real one. Any
    such slot is reported as UNVERIFIED rather than as a number.

Bands, and why the gap:

    B0  8 B - 8 KiB       latency, us      fixed fabric and RCCL cost
    B1  16 KiB - 1 MiB    latency, us      TP all-reduce and EP all-to-all: the
                                           band that decides this study
    B2  4 MiB - 128 MiB   bus bandwidth    chunked prefill
    B3  256 MiB - 2 GiB   bus bandwidth    bulk; kept for continuity with job
                                           21791400, not in the promotion scalar

1-4 MiB is measured but deliberately left out of every band: RCCL changes protocol
somewhere in there, and a bandwidth averaged across a regime change describes
neither regime. The full curve still shows it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import stats

BANDS = [
    ("B0 floor", 8, 8 * 1024, "latency"),
    ("B1 decode", 16 * 1024, 1024 * 1024, "latency"),
    ("B2 prefill", 4 * 1024**2, 128 * 1024**2, "bandwidth"),
    ("B3 bulk", 256 * 1024**2, 2 * 1024**3, "bandwidth"),
]

BUS_FACTOR = {
    "all_reduce": lambda n: 2.0 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_to_all": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
}

PROMOTE, HOLD, INERT, UNDERPOWERED, KILL = (
    "promote", "hold", "inert", "underpowered", "kill")


def verdict(gain_pct: float, half_pct: float) -> str:
    """The thresholds pre-registered in FINDINGS.md before the first sbatch.

    The 3% floor is hard regardless of significance: across a dozen screened
    factors a 2% "significant" result is mostly multiplicity, and a 2%
    microbenchmark gain cannot survive dilution into tok/s, so it could never
    clear the end-to-end acceptance rule and is unshippable either way.
    """
    if gain_pct <= -5.0:
        return KILL
    if gain_pct >= 5.0 and gain_pct - half_pct > 0:
        return PROMOTE
    if gain_pct >= 3.0:
        return HOLD
    if half_pct < 3.0:
        return INERT
    return UNDERPOWERED


def load_slots(results_dirs: list[str]) -> list[dict]:
    """One entry per (job, variant), aggregated across its ranks.

    Times are aggregated by **max across ranks**, not mean: a collective is only as
    fast as its slowest participant, which is the same argument summarize_sweep.py
    makes for taking the slowest rank's phase.
    """
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for results_dir in results_dirs:
        for path in sorted(glob.glob(os.path.join(results_dir, "*_rank*.json"))):
            with open(path, encoding="utf-8") as handle:
                rep = json.load(handle)
            # Position is part of the key, not just the name: the sentinel runs
            # twice per block and the two runs are the drift measurement.
            position = int(((rep.get("block") or {}).get("position")) or 0)
            by_slot[(rep.get("job_id"), rep.get("variant"), position)].append(rep)

    slots = []
    for (job_id, variant, position), ranks in sorted(by_slot.items()):
        head = ranks[0]
        block = head.get("block") or {}
        points: dict[tuple, dict] = {}
        for rank_rep in ranks:
            for row in rank_rep.get("rows", []):
                if "samples_seconds" not in row or not row["samples_seconds"]:
                    continue
                key = (row["op"], row["nominal_bytes"], row["dtype"])
                rep_t = stats.percentile(row["samples_seconds"], 0.5)
                p95_t = stats.percentile(row["samples_seconds"], 0.95)
                burst = row.get("burst_samples_seconds")
                burst_t = stats.percentile(burst, 0.5) if burst else None
                acc = points.setdefault(key, {"bytes": row["bytes"], "rep_s": 0.0,
                                              "p95_s": 0.0, "burst_s": None, "ranks": 0})
                acc["ranks"] += 1
                acc["rep_s"] = max(acc["rep_s"], rep_t)
                acc["p95_s"] = max(acc["p95_s"], p95_t)
                if burst_t is not None:
                    acc["burst_s"] = burst_t if acc["burst_s"] is None else max(acc["burst_s"], burst_t)

        requested = head.get("env_requested") or {}
        observed = head.get("env_observed") or {}
        missing = {k: v for k, v in requested.items() if observed.get(k) != v}

        slots.append({
            "job_id": job_id,
            "variant": variant,
            "role": block.get("role") or "design",
            "position": position,
            "block_id": block.get("id"),
            "replicate": block.get("replicate"),
            "stage": block.get("stage"),
            "world_size": head.get("world_size"),
            "nodes": head.get("nodes"),
            "nodelist": head.get("nodelist_raw"),
            "partition": head.get("partition"),
            "n_ranks": len(ranks),
            "points": points,
            "env_mismatch": missing,
            "evidence": head.get("effect_evidence") or {},
        })
    return slots


def band_metric(points: dict, band: tuple, prefer_burst: bool = True) -> dict | None:
    """One number per band: geometric mean over the points inside it.

    Geometric, because the points are a doubling ladder and an arithmetic mean over
    a band is simply the largest size wearing a disguise.
    """
    _, lo, hi, kind = band
    times, p95s, busbws = [], [], []
    for (op, nominal, dtype), point in points.items():
        if dtype != "bf16" or not (lo <= nominal <= hi):
            continue
        seconds = point["burst_s"] if (prefer_burst and point["burst_s"]) else point["rep_s"]
        if not seconds or seconds <= 0:
            continue
        times.append(seconds)
        p95s.append(point["p95_s"])
        world = point.get("world", 0)
        busbws.append(point["bytes"] / seconds / 1e9 * BUS_FACTOR[op](world) if world else
                      point["bytes"] / seconds / 1e9)
    if not times:
        return None
    if kind == "latency":
        return {"kind": kind, "value_us": stats.geomean(times) * 1e6,
                "p95_us": stats.geomean(p95s) * 1e6, "n_points": len(times),
                "log": math.log(stats.geomean(times))}
    return {"kind": kind, "value_gbps": stats.geomean(busbws), "n_points": len(times),
            "log": math.log(stats.geomean(busbws))}


def attach_world(slots: list[dict]) -> None:
    for slot in slots:
        for point in slot["points"].values():
            point["world"] = slot["world_size"] or 0


def promotion_scalar(points: dict, cfg: dict) -> float | None:
    """Predicted collective milliseconds per decode step at the shipped config.

    A prediction, not a summary. Combined with the collective fraction of a decode
    step it says how much end-to-end gain to expect, which turns Stage 5 from "did
    it get faster?" into "did it get faster by the predicted amount?".

    It is a lower bound on in-serving cost and so an upper bound on the achievable
    gain: a tight loop entered from a barrier has perfectly synchronised ranks, no
    compute to overlap with and no HBM contention.
    """
    def lookup(op: str, nbytes: int) -> float | None:
        point = points.get((op, nbytes, "bf16"))
        if not point:
            return None
        return point["burst_s"] or point["rep_s"]

    t_ar = lookup("all_reduce", cfg["decode_bytes"])
    t_a2a = lookup("all_to_all", cfg["a2a_bytes"])
    if t_ar is None:
        return None
    total = cfg["n_allreduce"] * t_ar
    if t_a2a is not None:
        total += cfg["n_alltoall"] * t_a2a
    return total * 1e3


def paired_gains(slots: list[dict], metric, higher_is_better: bool) -> dict:
    """Per variant, the log gain against the sentinel mean of its own allocation."""
    by_job = defaultdict(list)
    for slot in slots:
        by_job[slot["job_id"]].append(slot)

    gains: dict[str, list[float]] = defaultdict(list)
    for job_slots in by_job.values():
        refs = [metric(s) for s in job_slots if s["role"] == "sentinel"]
        refs = [math.log(v) for v in refs if v and v > 0]
        if not refs:
            continue
        ref_log = sum(refs) / len(refs)
        for slot in job_slots:
            if slot["role"] == "sentinel":
                continue
            value = metric(slot)
            if not value or value <= 0:
                continue
            delta = math.log(value) - ref_log
            gains[slot["variant"]].append(delta if higher_is_better else -delta)
    return gains


def control_report(slots: list[dict], cv_pos_pct: float | None, metric) -> list[str]:
    """The three controls. Any failure voids the allocation or the stage."""
    lines = ["## Controls\n"]
    by_job = defaultdict(list)
    for slot in slots:
        by_job[slot["job_id"]].append(slot)

    lines.append("| job | control | observed | rule | verdict |")
    lines.append("| --- | --- | --- | --- | --- |")
    for job_id, job_slots in sorted(by_job.items()):
        sentinels = sorted((s for s in job_slots if s["role"] == "sentinel"),
                           key=lambda s: s["position"])
        if len(sentinels) >= 2:
            first, last = metric(sentinels[0]), metric(sentinels[-1])
            if first and last:
                drift = abs(math.log(last / first)) * 100
                limit = 3 * cv_pos_pct if cv_pos_pct else None
                ok = "-" if limit is None else ("PASS" if drift <= limit else "**DISCARD**")
                rule = f"<= 3 sigma_pos = {limit:.1f}%" if limit else "needs Stage 0 CV_pos"
                lines.append(f"| `{job_id}` | sentinel drift | {drift:.2f}% | {rule} | {ok} |")

        for slot in job_slots:
            if slot["role"] == "sham":
                gains = paired_gains(job_slots, metric, False).get(slot["variant"], [])
                if gains:
                    pct = abs(stats.as_percent(gains[0]))
                    ok = "PASS" if pct < 3.0 else "**STAGE SUSPECT**"
                    lines.append(f"| `{job_id}` | sham (`{slot['variant']}`) | {pct:.2f}% | "
                                 f"< 3% or every positive in this stage is suspect | {ok} |")
            if slot["variant"] == "cap4":
                band = next(b for b in BANDS if b[0].startswith("B3"))
                value = band_metric(slot["points"], band)
                ref = [band_metric(s["points"], band) for s in job_slots if s["role"] == "sentinel"]
                ref = [r for r in ref if r]
                if value and ref:
                    loss = (value["value_gbps"] / stats.geomean([r["value_gbps"] for r in ref]) - 1) * 100
                    ok = "PASS" if loss < -20 else "**HARNESS NOT APPLYING ENV -- STOP**"
                    lines.append(f"| `{job_id}` | known-sign `cap4` | {loss:.1f}% B3 | "
                                 f"must be strongly negative | {ok} |")

    mismatched = [s for s in slots if s["env_mismatch"]]
    if mismatched:
        lines.append("")
        lines.append("> **UNVERIFIED — environment did not reach the ranks.** A knob that never "
                     "arrived produces a clean null indistinguishable from a real one, so these "
                     "slots carry no verdict:\n")
        for slot in mismatched:
            lines.append(f"> - `{slot['variant']}` (job `{slot['job_id']}`): "
                         f"{json.dumps(slot['env_mismatch'], sort_keys=True)}")
    else:
        lines.append("")
        lines.append("Every slot's requested environment was observed in rank 0. "
                     "(`env_requested` vs `env_observed`.)")
    return lines


def noise_report(slots: list[dict], metric) -> list[str]:
    """Stage 0. The only output that matters is the MDE the rest of the study buys."""
    by_job: dict[str, list[dict]] = defaultdict(list)
    for slot in slots:
        by_job[slot["job_id"]].append(slot)

    blocks, positions, values = [], [], []
    for job_slots in by_job.values():
        ordered = sorted(job_slots, key=lambda s: s["position"])
        logs = []
        for slot in ordered:
            value = metric(slot)
            if value and value > 0:
                logs.append(math.log(value))
                positions.append(slot["position"])
                values.append(math.log(value))
        if logs:
            blocks.append(logs)

    comp = stats.variance_components(blocks)
    lines = ["## Stage 0 — noise floor\n"]
    if "error" in comp:
        lines.append(f"> {comp['error']}\n")
        return lines

    # Position 1 pays one-time costs the later slots do not. A cold start is a step
    # and drift is a slope; estimating them together is how the five false findings
    # happened, so the slope is fitted on positions 2+ and the step reported apart.
    later = [(p, v) for p, v in zip(positions, values) if p >= 2]
    slope_fit = stats.ols_slope([p for p, _ in later], [v for _, v in later]) if len(later) >= 3 else None
    slope_pct = stats.as_percent(slope_fit[0]) if slope_fit else None

    firsts = [v for p, v in zip(positions, values) if p == 1]
    step_pct = None
    if firsts and slope_fit:
        predicted = slope_fit[0] * 1 + slope_fit[1]
        step_pct = stats.as_percent(sum(firsts) / len(firsts) - predicted)

    lines.append("| component | CV | what it is |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| allocation | {comp['cv_alloc_pct']:.2f}% | Dragonfly span, which nodes, "
                 "who else is on the switch — **cancelled by pairing** |")
    lines.append(f"| position | {comp['cv_pos_pct']:.2f}% | slot-to-slot inside one allocation "
                 "— **this is what sets the MDE** |")
    lines.append("")
    lines.append(f"- Position slope, fitted on slots 2+: "
                 f"**{slope_pct:+.2f}%/slot**" if slope_pct is not None else
                 "- Position slope: not enough slots to fit")
    if step_pct is not None:
        lines.append(f"- Slot-1 step, estimated apart from the slope: **{step_pct:+.2f}%**")
    lines.append(f"- Allocations: {comp['n_alloc']}, slots: {comp['n_slots']}")
    lines.append("")
    lines.append("| allocations per arm | MDE |")
    lines.append("| --- | --- |")
    for k in (2, 3, 4, 6, 12):
        lines.append(f"| {k} | {stats.mde_percent(comp['cv_pos_pct'], k):.2f}% |")
    lines.append("")
    if slope_pct is not None:
        if abs(slope_pct) < 1.0:
            lines.append("> Position slope is under 1%/slot, so variants may share an allocation "
                         "as long as order is randomised and the sentinel gate holds. The "
                         "shared-allocation design in `run_bandwidth.sh` was sound for the effect "
                         "sizes it measured.")
        else:
            lines.append(f"> Position slope is {slope_pct:+.2f}%/slot, large enough to swamp the "
                         "3-10% effects this study is looking for. Either raise the replicate "
                         "count until the randomisation averages it out, or go to one variant per "
                         "allocation and re-cost the study.")
    return lines


def screen_report(slots: list[dict], block_files: list[str], metric) -> list[str]:
    """Stage 2. Main effects from the PB design, with Lenth's PSE for the error."""
    levels: dict[str, dict[str, int]] = {}
    factors: list[str] = []
    for path in block_files:
        with open(path, encoding="utf-8") as handle:
            block = json.load(handle)
        for slot in block.get("slots", []):
            if slot.get("role") == "design" and isinstance(slot.get("levels"), dict):
                if all(isinstance(v, int) for v in slot["levels"].values()):
                    levels[slot["name"]] = slot["levels"]
                    for key in slot["levels"]:
                        if key not in factors:
                            factors.append(key)
    if not levels:
        return []

    gains = paired_gains(slots, metric, False)
    effects, rows = {}, []
    for factor in factors:
        high = [g for name, gs in gains.items() if levels.get(name, {}).get(factor) == 1 for g in gs]
        low = [g for name, gs in gains.items() if levels.get(name, {}).get(factor) == -1 for g in gs]
        if not high or not low:
            continue
        effects[factor] = sum(high) / len(high) - sum(low) / len(low)
        rows.append((factor, effects[factor], len(high), len(low)))

    if not rows:
        return []

    pse = stats.lenth_pse(list(effects.values()))
    pvalues = [stats.normal_sf(abs(e) / pse) * 2 if pse else 1.0 for _, e, _, _ in rows]
    keep = stats.benjamini_hochberg(pvalues, q=0.10)

    lines = ["## Stage 2 — screen (Plackett-Burman, main effects)\n",
             "Each effect is the mean paired gain at the high level minus the low level, "
             "every run already divided by the sentinel of its own allocation. "
             f"Error from Lenth's pseudo standard error (PSE = {stats.as_percent(pse):.2f}%) "
             "with Benjamini-Hochberg at q = 0.10.\n",
             "| factor | effect | high/low runs | p | survives BH | verdict |",
             "| --- | --- | --- | --- | --- | --- |"]
    for (factor, effect, n_high, n_low), pvalue, survives in sorted(
            zip(rows, pvalues, keep), key=lambda t: -abs(t[0][1])):
        pct = stats.as_percent(effect)
        half = stats.as_percent(2 * pse) if pse else float("inf")
        lines.append(f"| `{factor}` | {pct:+.2f}% | {n_high}/{n_low} | {pvalue:.3f} | "
                     f"{'yes' if survives else 'no'} | {verdict(pct, half)} |")
    lines.append("")
    lines.append("> The top effect here is biased upward by selection. Stage 4 re-estimates it "
                 "on fresh allocations, and `env_tuned.sh` quotes the Stage-4 number.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", action="append", default=[],
                        help="a job's results directory; repeat to pool a whole stage")
    parser.add_argument("--results-root", help="glob <root>/job_* instead of listing them")
    parser.add_argument("--out", help="markdown path (default: <first results dir>/tuning.md)")
    parser.add_argument("--cv-pos-pct", type=float,
                        help="position CV from Stage 0; without it the sentinel gate has no bound")
    parser.add_argument("--band", default="B1 decode", help="band the verdict column uses")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="analyse blocks that ran fewer slots than their design")
    parser.add_argument("--decode-bytes", type=int, default=917504,
                        help="hidden x max_num_seqs x 2 = 7168 x 64 x 2")
    parser.add_argument("--a2a-bytes", type=int, default=458752,
                        help="tokens x topk x hidden x 2 / EP")
    parser.add_argument("--n-allreduce", type=int, default=61, help="2 x n_layers / pp")
    parser.add_argument("--n-alltoall", type=int, default=60, help="2 x n_moe_layers / pp")
    args = parser.parse_args()

    dirs = list(args.results_dir)
    if args.results_root:
        dirs.extend(sorted(glob.glob(os.path.join(args.results_root, "job_*"))))
    if not dirs:
        dirs = [os.environ.get("RESULTS_DIR", "results")]

    # run_block.sh drops this marker when it ran fewer slots than its design. Such a
    # job is not a block: its sentinels may be missing and its randomisation is
    # truncated, so pooling it would quietly bias everything it touches.
    incomplete = [d for d in dirs if os.path.exists(os.path.join(d, "INCOMPLETE_BLOCK"))]
    if incomplete and not args.allow_incomplete:
        print("refusing to analyse incomplete blocks (pass --allow-incomplete to override):")
        for d in incomplete:
            print(f"  {d}")
        return 3

    slots = load_slots(dirs)
    if not slots:
        print(f"no *_rank*.json under {dirs}")
        return 2
    discarded = [s for s in slots if s["role"] == "warmup"]
    slots = [s for s in slots if s["role"] != "warmup"]
    attach_world(slots)

    cfg = {"decode_bytes": args.decode_bytes, "a2a_bytes": args.a2a_bytes,
           "n_allreduce": args.n_allreduce, "n_alltoall": args.n_alltoall}
    band = next((b for b in BANDS if b[0] == args.band), BANDS[1])

    def band_value(slot):
        got = band_metric(slot["points"], band)
        if not got:
            return None
        return got["value_us"] if got["kind"] == "latency" else got["value_gbps"]

    def ps_value(slot):
        return promotion_scalar(slot["points"], cfg)

    stages = sorted({s["stage"] for s in slots if s["stage"]})
    jobs = sorted({s["job_id"] for s in slots})
    head = slots[0]
    lines = [f"# RCCL tuning — stage {'/'.join(stages) or '?'}\n",
             f"Jobs `{'`, `'.join(str(j) for j in jobs)}`. "
             f"{head['nodes']} node(s), world {head['world_size']}, "
             f"partition `{head['partition']}`. {len(slots)} slots.\n",
             f"Nodelists: {', '.join(sorted({str(s['nodelist']) for s in slots}))}\n"]

    if discarded:
        lines.append(f"{len(discarded)} cold warm-up slot(s) run and discarded, so both "
                     "sentinels in every block are warm and the drift gate measures drift "
                     "rather than cold start.\n")
    lines += control_report(slots, args.cv_pos_pct, band_value)
    lines.append("")

    if any(s["stage"] == "0" for s in slots):
        lines += noise_report(slots, band_value)
        lines.append("")

    block_files = [p for d in dirs for p in glob.glob(os.path.join(d, "block.json"))]
    screen = screen_report(slots, block_files, ps_value)
    if screen:
        lines += screen
        lines.append("")

    # Slots whose environment never arrived are dropped from the verdict, but the
    # sentinels must stay in: they are what every gain is measured against.
    unverified = {s["variant"] for s in slots if s["env_mismatch"]}
    measurable = [s for s in slots if s["role"] != "sentinel" and s["variant"] not in unverified]
    if measurable:
        lines.append(f"## Paired gains vs the sentinel — {band[0]}, and the promotion scalar\n")
        lines.append("Positive is better for every column: a latency band is reported as the "
                     "reduction in time, a bandwidth band as the increase in bandwidth. "
                     "`PS` is predicted collective ms per decode step "
                     f"({cfg['n_allreduce']} all-reduce of {cfg['decode_bytes'] // 1024} KiB + "
                     f"{cfg['n_alltoall']} all-to-all of {cfg['a2a_bytes'] // 1024} KiB).\n")
        band_gains = {k: v for k, v in paired_gains(slots, band_value,
                                                    band[3] == "bandwidth").items()
                      if k not in unverified}
        ps_gains = paired_gains(slots, ps_value, False)
        lines.append(f"| variant | {band[0]} gain | PS gain | n | verdict |")
        lines.append("| --- | --- | --- | --- | --- |")
        for variant in sorted(band_gains, key=lambda v: -(stats.mean_ci(ps_gains.get(v) or [0]) or (0, 0))[0]):
            bstat = stats.mean_ci(band_gains[variant])
            pstat = stats.mean_ci(ps_gains.get(variant, []))
            if not bstat:
                continue
            bpct, bhalf = stats.as_percent(bstat[0]), stats.as_percent(bstat[1])
            if pstat:
                ppct, phalf = stats.as_percent(pstat[0]), stats.as_percent(pstat[1])
                cell = f"{ppct:+.2f}% ± {phalf:.2f}"
                call = verdict(ppct, phalf)
            else:
                cell, call = "-", "-"
            lines.append(f"| `{variant}` | {bpct:+.2f}% ± {bhalf:.2f} | {cell} | "
                         f"{len(band_gains[variant])} | {call} |")
        lines.append("")

    lines.append("## Bands, absolute\n")
    lines.append("| variant | " + " | ".join(b[0] for b in BANDS) + " |")
    lines.append("| " + " | ".join(["---"] * (len(BANDS) + 1)) + " |")
    for slot in sorted(slots, key=lambda s: (s["job_id"] or "", s["position"])):
        cells = [f"`{slot['variant']}`@{slot['position']}"]
        for b in BANDS:
            got = band_metric(slot["points"], b)
            if not got:
                cells.append("-")
            elif got["kind"] == "latency":
                cells.append(f"{got['value_us']:.1f} / {got['p95_us']:.1f} us")
            else:
                cells.append(f"{got['value_gbps']:.1f} GB/s")
        lines.append("| " + " | ".join(cells) + " |")

    out_path = args.out or os.path.join(dirs[0], "tuning.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
