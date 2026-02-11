#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
from typing import List, Dict


NAME_RE = re.compile(r"summary_r(\d+)_c(\d+)_t(\d+)\.json$")


def load_rows(job_dir: str) -> List[Dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(job_dir, "summary_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = NAME_RE.search(os.path.basename(path))
        requests = int(m.group(1)) if m else data.get("requests_total")
        concurrency = int(m.group(2)) if m else data.get("concurrency")
        max_tokens = int(m.group(3)) if m else data.get("max_tokens")

        rows.append(
            {
                "file": os.path.basename(path),
                "requests": requests,
                "concurrency": concurrency,
                "max_tokens": max_tokens,
                "requests_total": int(data.get("requests_total", 0)),
                "requests_ok": int(data.get("requests_ok", 0)),
                "requests_failed": int(data.get("requests_failed", 0)),
                "p50": float(data.get("latency_p50_s", 0.0)),
                "p95": float(data.get("latency_p95_s", 0.0)),
                "p99": float(data.get("latency_p99_s", 0.0)),
                "req_s": float(data.get("throughput_req_s", 0.0)),
                "tok_s": float(data.get("throughput_tokens_s", 0.0)),
                "ctok_s": float(data.get("throughput_completion_tokens_s", 0.0)),
            }
        )
    return rows


def print_table(rows: List[Dict]) -> None:
    header = (
        "file,requests,concurrency,max_tokens,ok,failed,"
        "p50_s,p95_s,throughput_req_s,throughput_tokens_s,throughput_completion_tokens_s"
    )
    print(header)
    for r in rows:
        print(
            f"{r['file']},{r['requests']},{r['concurrency']},{r['max_tokens']},"
            f"{r['requests_ok']},{r['requests_failed']},"
            f"{r['p50']:.3f},{r['p95']:.3f},{r['req_s']:.3f},{r['tok_s']:.3f},{r['ctok_s']:.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark summary_*.json files")
    parser.add_argument("--job-dir", required=True, help="Path to benchmarks/results/job_<jobid>")
    parser.add_argument(
        "--sort-by",
        default="ctok_s",
        choices=["ctok_s", "tok_s", "req_s", "p95"],
        help="Metric to sort by",
    )
    args = parser.parse_args()

    rows = load_rows(args.job_dir)
    if not rows:
        print(f"No summary_*.json files found in {args.job_dir}", file=sys.stderr)
        return 2

    reverse = args.sort_by != "p95"
    rows = sorted(rows, key=lambda x: x[args.sort_by], reverse=reverse)
    print_table(rows)

    no_fail = [r for r in rows if r["requests_failed"] == 0]
    if no_fail:
        best = no_fail[0]
        print("\nBest (0 failures):")
        print(
            f"file={best['file']} concurrency={best['concurrency']} max_tokens={best['max_tokens']} "
            f"p95={best['p95']:.3f}s throughput_completion_tokens_s={best['ctok_s']:.3f}"
        )
    else:
        print("\nNo zero-failure run found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

