#!/usr/bin/env python3
"""Stage 1: prove each knob reaches RCCL, before spending allocations on measuring it.

A knob that silently does nothing produces a clean null that is indistinguishable
from a real one. That is the failure mode this whole file exists to prevent, and
the rule it feeds is enforced in analyze.py:

    A null result on an unverified knob is not a null result.

Three layers, cheapest first:

  1. **Environment.** Did the variable reach the rank at all? `collective_profile.py`
     records `env_requested` against `env_observed`, and analyze.py already refuses a
     verdict on any mismatch. singularity strips most of the environment unless it is
     passed through, and no version of this harness had ever checked.
  2. **Log assertion.** Did RCCL or libfabric act on it? Parsed here from a
     `SLOT_DEBUG=1` run. This is why Stage 1 is a separate 2-node job: per-rank INFO
     logging measurably changes the timings it would otherwise be explaining.
  3. **Behavioural.** For knobs with no log line and no proxy, there is nothing to
     assert, and the honest answer is `no-proxy` rather than a number.

Usage, after a SLOT_DEBUG=1 block has run:

    python3 verify_effect.py --results-dir repro/rccl_tuning_gfx90a/results/job_<id>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

# knob -> (pattern over the INFO log, what a hit means).
# Absence is never read as "the knob did nothing"; it is read as "unproven".
LOG_ASSERTIONS = {
    "NCCL_MAX_NCHANNELS": (r"(\d+)\s+coll channels", "RCCL reported its channel count"),
    "NCCL_MIN_NCHANNELS": (r"(\d+)\s+coll channels", "RCCL reported its channel count"),
    "NCCL_NCHANNELS_PER_NET_PEER": (r"(\d+)\s+coll channels", "RCCL reported its channel count"),
    "NCCL_NET_GDR_READ": (r"use ring PXN \d+ GDR (\d+)", "RCCL reported the ring's GDR state"),
}

# RCCL refuses the knob and says so. A match here is a knob PROVEN INERT, which is a
# result worth shipping, not a verification failure -- so it is kept apart from the
# patterns above, which prove the opposite.
REFUSAL_ASSERTIONS = {
    "RCCL_MSCCLPP_ENABLE": (
        r"Cannot enable MSCCL\+\+; environment is not MSCCL compatible",
        "RCCL refused to enable MSCCL++ on this container/gfx90a"),
    "RCCL_MSCCL_FORCE_ENABLE": (
        r"Cannot enable MSCCL",
        "RCCL refused to enable MSCCL on this container/gfx90a"),
}

# Knobs with no userspace evidence at all. Named here so they are reported as
# no-proxy rather than quietly appearing to have been checked.
NO_PROXY = {
    "HSA_NO_SCRATCH_RECLAIM":
        "read by the HSA runtime, not RCCL: no log line and no fd to inspect. The "
        "container is pinned at ROCm 7.0 while AMD documents this knob for ROCm 7.13+, "
        "so record the ROCm version and treat the B0/B1 latency A/B as the only evidence.",
    "NCCL_IGNORE_CPU_AFFINITY":
        "changes which CPUs RCCL's helper threads may run on; nothing reports it back.",
    # Checked against a real INIT,TUNING capture (job 22119061): RCCL prints no line
    # for either of these. Absence of a line is not absence of an effect.
    "NCCL_BUFFSIZE":
        "RCCL logs no buffer-size line at INIT or TUNING; verified against job 22119061.",
    "NCCL_CROSS_NIC":
        "RCCL logs no cross-NIC line at INIT or TUNING; verified against job 22119061.",
    # The Algorithm/Protocol lines in the INIT trace are the header of the tuner's
    # COST TABLE -- every algorithm and protocol RCCL knows about, printed whether or
    # not it is chosen. Matching them proves nothing about what was selected, which is
    # what an earlier revision of this file wrongly asserted.
    "NCCL_ALGO":
        "the Algorithm line is the tuner cost-table header, not a selection record; "
        "RCCL does not log which algorithm it chose.",
    "NCCL_PROTO":
        "the Protocol line is the tuner cost-table header, not a selection record. "
        "NCCL_PROTO=LL128 is separately PROVEN to take effect by a hard failure: RCCL "
        "2.26.6 has no LL128 path for bf16 all-reduce and errors out (job 22118353).",
    # libfabric writes to stderr under FI_LOG_LEVEL, never to NCCL_DEBUG_FILE, so the
    # NCCL capture can never contain them. Discriminate behaviourally instead.
    "FI_CXI_RDZV_PROTO":
        "libfabric logs to stderr, not NCCL_DEBUG_FILE, and falls back silently when "
        "the driver lacks rdzv_get_en=0. Needs a behavioural signature.",
    "FI_CXI_RDZV_THRESHOLD": "libfabric logs to stderr, not NCCL_DEBUG_FILE.",
    "FI_CXI_RDZV_EAGER_SIZE": "libfabric logs to stderr, not NCCL_DEBUG_FILE.",
    "FI_CXI_RDZV_GET_MIN": "libfabric logs to stderr, not NCCL_DEBUG_FILE.",
    "FI_CXI_DEFAULT_TX_SIZE": "libfabric logs to stderr, not NCCL_DEBUG_FILE.",
}


def read_logs(results_dir: str) -> dict[str, str]:
    """One concatenated INFO capture per variant."""
    logs: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "debug_*_rank*.log"))):
        match = re.match(r"debug_(.+)_rank\d+\.log$", os.path.basename(path))
        if not match:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                logs.setdefault(match.group(1), []).append(handle.read())
        except OSError:
            continue
    return {name: "\n".join(parts) for name, parts in logs.items()}


def read_slots(results_dir: str) -> dict[str, dict]:
    """Rank 0 of each slot, for the environment and fd evidence."""
    slots = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_rank0000.json"))):
        with open(path, encoding="utf-8") as handle:
            rep = json.load(handle)
        slots[rep.get("variant")] = rep
    return slots


def verify(slot: dict, log: str) -> dict:
    """Per knob: true, false, or 'no-proxy', with the evidence that decided it."""
    requested = slot.get("env_requested") or {}
    observed = slot.get("env_observed") or {}
    evidence = slot.get("effect_evidence") or {}
    out = {}

    for key, value in sorted(requested.items()):
        record: dict = {"requested": value, "observed": observed.get(key)}

        if observed.get(key) != value:
            record.update(verified=False, layer="environment",
                          detail="the variable never reached the rank")
            out[key] = record
            continue

        if key == "FI_MR_CACHE_MONITOR":
            # Behavioural, reusing the trick that settled the monitor question in the
            # bug study: userfaultfd shows as an fd, kdreg2 as its device node.
            uffd, kdreg2 = evidence.get("userfaultfd_fds", 0), evidence.get("kdreg2_open", False)
            expected = {"userfaultfd": uffd > 0, "kdreg2": bool(kdreg2),
                        "memhooks": uffd == 0 and not kdreg2,
                        "disabled": uffd == 0 and not kdreg2}.get(value)
            record.update(verified=bool(expected), layer="behavioural",
                          detail=f"userfaultfd_fds={uffd}, kdreg2_open={kdreg2}")
            out[key] = record
            continue

        if key in NO_PROXY:
            record.update(verified="no-proxy", layer="none", detail=NO_PROXY[key])
            out[key] = record
            continue

        refusal = REFUSAL_ASSERTIONS.get(key)
        if refusal and log and re.search(refusal[0], log):
            record.update(verified="refused", layer="log", detail=refusal[1])
            out[key] = record
            continue

        pattern, meaning = LOG_ASSERTIONS.get(key, (None, None))
        if pattern is None:
            record.update(verified="no-proxy", layer="none",
                          detail="no assertion defined for this knob")
        elif not log:
            record.update(verified=False, layer="log",
                          detail="no INFO capture for this slot; re-run with SLOT_DEBUG=1")
        else:
            hits = re.findall(pattern, log)
            record.update(verified=bool(hits), layer="log",
                          detail=f"{meaning}: {sorted(set(hits))[:4]}" if hits
                          else f"no line matching /{pattern}/")
        out[key] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "results"))
    args = parser.parse_args()

    logs, slots = read_logs(args.results_dir), read_slots(args.results_dir)
    if not slots:
        print(f"no *_rank0000.json under {args.results_dir}")
        return 2

    report = {name: verify(slot, logs.get(name, "")) for name, slot in sorted(slots.items())}
    out_path = os.path.join(args.results_dir, "knob_verified.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    lines = ["# Stage 1 — does each knob take effect?\n",
             "`true` means something outside this harness confirmed the knob was acted on. "
             "`no-proxy` means nothing can confirm it from userspace, which is a limitation "
             "to state, not a null result to report. `false` means the knob is dropped from "
             "the design with this job id against it.\n",
             "| slot | knob | verified | layer | evidence |",
             "| --- | --- | --- | --- | --- |"]
    unverified, no_proxy, refused = [], [], []
    for name, knobs in report.items():
        for key, record in knobs.items():
            mark = {True: "yes", False: "**no**",
                    "refused": "**refused**"}.get(record["verified"], "`no-proxy`")
            lines.append(f"| `{name}` | `{key}` | {mark} | {record['layer']} | "
                         f"{record['detail']} |")
            if record["verified"] is False:
                unverified.append((name, key))
            elif record["verified"] == "refused":
                refused.append((name, key, record["detail"]))
            elif record["verified"] == "no-proxy":
                no_proxy.append((name, key))

    if refused:
        lines.append("\n## Proven inert -- drop from the design, and ship the finding\n")
        lines.append("> RCCL itself refused the knob. This is a result: a knob that cannot "
                     "be enabled on this hardware is worth more to the next person, with a "
                     "job id against it, than a column in a screen.\n")
        for name, key, detail in refused:
            lines.append(f"- `{key}` (slot `{name}`) — {detail}")

    if unverified:
        lines.append("\n## Drop these from the design\n")
        lines.append("> Each is recorded in FINDINGS.md against this job id. Carrying an "
                     "unverified knob into the screen would spend allocations producing a "
                     "null that means nothing.\n")
        for name, key in unverified:
            lines.append(f"- `{key}` (slot `{name}`)")
    if no_proxy:
        lines.append("\n## Measurable, but only behaviourally\n")
        for name, key in no_proxy:
            lines.append(f"- `{key}` (slot `{name}`) — needs a pre-declared expected "
                         "signature before it is measured")

    md_path = os.path.join(args.results_dir, "knob_verified.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}\nwrote {md_path}\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
