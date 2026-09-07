#!/usr/bin/env python3
"""Turn the bandwidth JSON into results/bandwidth.md -- the answer to "what does
capping channels cost?"

The headline number is the loss in the 128 MiB - 1 GiB band, because that is where
gradient all-reduce buckets in bandwidth-bound training sit. Small-message latency is
reported separately: capping channels typically costs little there and most at large
sizes, and collapsing the two into one average would answer the wrong question.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

TRAINING_BAND = (128 * 1024**2, 1024**3)
SMALL_BAND = (0, 1024**2)
BASELINE_HINTS = ("default_channels", "default", "baseline")


def human_bytes(size: int) -> str:
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= scale:
            return f"{size / scale:g} {unit}"
    return f"{size} B"


def variant_order(name: str) -> tuple[int, int]:
    """Uncapped first, then descending channel count, so the table reads as a ladder."""
    if name in BASELINE_HINTS:
        return (0, 0)
    digits = "".join(c for c in name if c.isdigit())
    return (1, -int(digits)) if digits else (2, 0)


def load(results_dir: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "bandwidth_*.json"))):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        out[report.get("variant", os.path.basename(path))] = report
    return out


def band_mean(rows: list[dict], op: str, lo: int, hi: int, field: str) -> float | None:
    values = [r[field] for r in rows
              if r.get("op") == op and "error" not in r and lo <= r["bytes"] <= hi]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = parser.parse_args()

    reports = load(args.results_dir)
    if not reports:
        print(f"no bandwidth_*.json in {args.results_dir}")
        return 2

    reports = dict(sorted(reports.items(), key=lambda kv: variant_order(kv[0])))
    baseline_name = next((n for n in reports if n in BASELINE_HINTS), None)
    ops = sorted({r["op"] for rep in reports.values() for r in rep["rows"]})
    any_rep = next(iter(reports.values()))

    lines: list[str] = ["# Collective bandwidth vs channel cap\n"]
    lines.append(
        f"Job `{any_rep['job_id']}`, {any_rep['nodes']} node(s), world size "
        f"{any_rep['world_size']}. Median of {any_rep['reps']} reps, "
        f"{any_rep['warmup']} discarded as warmup. Bus bandwidth follows the "
        f"nccl-tests convention.\n"
    )
    if baseline_name is None:
        lines.append(
            "> No uncapped run present, so the cost column cannot be computed. Re-run "
            "including the `default_channels` variant.\n"
        )

    lines.append(f"## Training band ({TRAINING_BAND[0] // 1024**2} MiB - "
                 f"{TRAINING_BAND[1] // 1024**2} MiB), mean bus bandwidth GB/s\n")
    header = ["variant"] + ops + [
        f"bandwidth lost vs `{baseline_name}`" if baseline_name else "bandwidth lost"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    base_band = {}
    if baseline_name:
        base_band = {op: band_mean(reports[baseline_name]["rows"], op, *TRAINING_BAND, "busbw_gbps")
                     for op in ops}

    for name, rep in reports.items():
        cells = [f"`{name}`"]
        losses = []
        for op in ops:
            value = band_mean(rep["rows"], op, *TRAINING_BAND, "busbw_gbps")
            cells.append(f"{value:.1f}" if value is not None else "-")
            ref = base_band.get(op)
            if ref and value is not None and ref > 0:
                losses.append((ref - value) / ref * 100.0)
        if losses and name != baseline_name:
            worst = max(losses)
            # Bold only a loss big enough to change a decision about a training job.
            cells.append(f"**{worst:.0f}%**" if worst >= 10 else f"{worst:.0f}%")
        else:
            cells.append("reference" if name == baseline_name else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append(f"\n## Small messages (<= {SMALL_BAND[1] // 1024} KiB), "
                 f"mean bus bandwidth GB/s\n")
    lines.append("| " + " | ".join(["variant"] + ops) + " |")
    lines.append("| " + " | ".join(["---"] * (len(ops) + 1)) + " |")
    for name, rep in reports.items():
        cells = [f"`{name}`"]
        for op in ops:
            value = band_mean(rep["rows"], op, *SMALL_BAND, "busbw_gbps")
            cells.append(f"{value:.2f}" if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Full curves (bus bandwidth GB/s)\n")
    for op in ops:
        lines.append(f"### {op}\n")
        sizes = sorted({r["bytes"] for rep in reports.values()
                        for r in rep["rows"] if r.get("op") == op and "error" not in r})
        lines.append("| size | " + " | ".join(f"`{n}`" for n in reports) + " |")
        lines.append("| " + " | ".join(["---"] * (len(reports) + 1)) + " |")
        for size in sizes:
            cells = [human_bytes(size)]
            for rep in reports.values():
                match = next((r for r in rep["rows"]
                              if r.get("op") == op and r.get("bytes") == size
                              and "error" not in r), None)
                cells.append(f"{match['busbw_gbps']:.2f}" if match else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    out_path = os.path.join(args.results_dir, "bandwidth.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"wrote {out_path}\n")
    for name, rep in reports.items():
        parts = []
        for op in ops:
            value = band_mean(rep["rows"], op, *TRAINING_BAND, "busbw_gbps")
            parts.append(f"{op}={value:.1f}" if value is not None else f"{op}=-")
        print(f"  {name:20s} training-band busbw GB/s: {'  '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
