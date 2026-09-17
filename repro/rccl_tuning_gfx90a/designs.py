#!/usr/bin/env python3
"""The experiment design, as a pure function of (stage, replicate, seed).

Nothing here touches Slurm, torch or the filesystem, so the whole design can be
printed, unit-tested and pre-registered on a laptop before a single GPU-hour is
spent. `submit_stage.sh` prints what this module returns into FINDINGS.md before
it submits; after that the design cannot be adjusted in light of the data.

Why Plackett-Burman at N=20 rather than the cheaper N=16
--------------------------------------------------------
A Hadamard matrix of order 16 built the usual (Sylvester) way is a *regular*
2^(15-11) fraction: every two-factor interaction is aliased 1:1 with some main
effect, so one strong interaction can masquerade completely as one factor. The
screen's whole justification is that it spreads interaction bias thinly instead.
PB-20 -- the genuine cyclic Plackett-Burman construction -- has the complex
partial aliasing that argument needs. It costs 4 extra runs per replicate
(~8 GPU-h over the whole screen) and buys 7 dummy columns of pure error
instead of 4. `test_designs.py` checks balance and orthogonality rather than
trusting the generator row transcribed below.
"""
from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass, field

# The guide's 8-task/node layout, from run_rccl_probe.sh:57. A bandwidth number
# taken without CPU-GPU affinity understates the machine, so this is the LOW
# (reference) level and dropping it is the HIGH level.
CPU_BIND_MASKS = (
    "0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,"
    "0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"
)
BIND_FLAG = f"--cpu-bind=v,mask_cpu={CPU_BIND_MASKS}"

# HPE's rendezvous group, moved as one factor. HPE ships them as a set and
# job 21838977 measured the whole non-monitor set inert for the hang; splitting
# them would cost four design columns for no decision anyone would make.
CXI_RDZV_SET = {
    "FI_CXI_RDZV_PROTO": "alt_read",
    "FI_CXI_RDZV_EAGER_SIZE": "0",
    "FI_CXI_RDZV_THRESHOLD": "0",
    "FI_CXI_RDZV_GET_MIN": "0",
    "FI_CXI_DEFAULT_TX_SIZE": "2048",
}

# Applied to every slot including the sentinel: the bug study's conclusion, which
# this study tunes on top of rather than re-deciding.
REF_ENV = {
    "FI_MR_CACHE_MONITOR": "userfaultfd",
}


@dataclass(frozen=True)
class Factor:
    key: str
    low: dict = field(default_factory=dict)
    high: dict = field(default_factory=dict)
    low_flags: str = BIND_FLAG
    high_flags: str = BIND_FLAG
    note: str = ""


# Order is fixed and load-bearing: it is the column assignment into PB-20, and
# changing it changes which interactions alias where. Append, never reorder.
#
# Two candidates were removed after Stage 1 (job 22119061) proved they cannot take
# effect on this machine, which is exactly what that stage is for:
#   mr_monitor (userfaultfd vs kdreg2) -- libfabric answers "kdreg2 monitor not
#     available" and falls back, so the high level is unreachable and the factor is
#     a constant. This also withdraws env_baseline.sh's claim that kdreg2 is an
#     equally valid fix.
#   mscclpp -- RCCL answers "Cannot enable MSCCL++; environment is not MSCCL
#     compatible". The knob is a no-op here.
FACTORS = [
    Factor("scratch_reclaim", {}, {"HSA_NO_SCRATCH_RECLAIM": "1"},
           note="AMD documents 5-10x small-message latency on gfx90a without it"),
    Factor("min_nchannels", {}, {"NCCL_MIN_NCHANNELS": "32"}),
    Factor("max_nchannels", {}, {"NCCL_MAX_NCHANNELS": "16"}),
    Factor("nchannels_per_peer", {}, {"NCCL_NCHANNELS_PER_NET_PEER": "2"}),
    Factor("buffsize", {}, {"NCCL_BUFFSIZE": "8388608"}, note="default is 4 MiB"),
    Factor("cross_nic", {}, {"NCCL_CROSS_NIC": "1"}, note="HPE recommends 1, untested on LUMI"),
    Factor("net_gdr_read", {}, {"NCCL_NET_GDR_READ": "1"}),
    Factor("ignore_cpu_affinity", {}, {"NCCL_IGNORE_CPU_AFFINITY": "1"}),
    Factor("cxi_rdzv", {}, dict(CXI_RDZV_SET),
           note="HPE's rendezvous set; THRESHOLD=0 forces every message through rendezvous"),
    Factor("cpu_bind", {}, {}, low_flags=BIND_FLAG, high_flags="",
           note="microbenchmark-only: the vLLM path runs 1 task/node where the mask is moot"),
]

def _all_managed_vars() -> list[str]:
    """Every key any slot in any stage can set.

    Derived from FACTORS alone this missed NCCL_ALGO and NCCL_PROTO, which only the
    Stage 1 checks and the Stage 3a grid set. run_block.sh therefore never unset
    them, they leaked from one slot into the next, and job 22117498 died with
    "no algorithm/protocol available for AllReduce with ncclBfloat16, NCCL_ALGO was
    set to Ring, NCCL_PROTO was set to LL128" -- a combination no single slot asked
    for. Walking the actual designs is self-maintaining; a new knob cannot be added
    to a stage without appearing here.
    """
    keys = set(REF_ENV) | {k for f in FACTORS for k in list(f.low) + list(f.high)}
    for build in (verify_blocks, screen_blocks,
                  lambda r, s: [algo_proto_block(r, s)],
                  lambda r, s: [channels_buffsize_block(r, s)],
                  lambda r, s: [min_nchannels_block(r, s)]):
        for block in build(1, 0):
            for slot in block["slots"]:
                keys.update(slot["env"])
    return sorted(keys)


# Populated at the bottom of the module, once every design function exists.
MANAGED_VARS: list[str] = []

# Plackett-Burman N=20, cyclic construction. 19 columns: FACTORS take the first
# len(FACTORS), the rest are dummies and carry the pure-error estimate.
PB20_GENERATOR = "++--++++-+-+----++-"


def pb20() -> list[list[int]]:
    """20 runs x 19 columns of +/-1. Rows 0..18 are cyclic shifts, row 19 all low."""
    row = [1 if c == "+" else -1 for c in PB20_GENERATOR]
    n = len(row)
    design = [[row[(j - i) % n] for j in range(n)] for i in range(n)]
    design.append([-1] * n)
    return design


def _rng(stage: str, replicate: int, seed: int) -> random.Random:
    """Deterministic and stable across interpreters -- unlike hash()."""
    key = f"{stage}|{replicate}|{seed}".encode()
    return random.Random(zlib.crc32(key))


def _slot(name: str, env: dict, flags: str, role: str, **extra) -> dict:
    merged = dict(REF_ENV)
    merged.update(env)
    return {"name": name, "env": merged, "srun_flags": flags, "role": role, **extra}


def _sentinel() -> dict:
    return _slot("ref", {}, BIND_FLAG, "sentinel")


def _sham(tag: str) -> dict:
    """The reference config wearing a different name. Any effect it shows is the
    study's own false-positive rate, measured rather than assumed."""
    return _slot(f"sham_{tag}", {}, BIND_FLAG, "sham")


def _cap4() -> dict:
    """Known-sign control: measured at -56% bus bandwidth in the bulk band
    (jobs 21791400, 21838863). If this does not come back clearly negative, the
    harness is not reaching the ranks with its environment and the stage is void."""
    return _slot("cap4", {"NCCL_MAX_NCHANNELS": "4"}, BIND_FLAG, "control")


def _warmup() -> dict:
    """Run first, measured, and thrown away.

    The first slot in an allocation pays one-time costs no later slot does: MIOpen
    compilation, Lustre first touch, a cold MR cache. The bug study measured that
    step at 290 s against 123 s for vLLM startup. If the opening sentinel wears it,
    log(S'/S) is reporting cold start rather than drift and the gate would discard
    every block. So the cold slot is a designated throwaway and both sentinels are
    warm. Stage 0 deliberately omits it, because quantifying that step is its job.
    """
    return _slot("warmup", {}, BIND_FLAG, "warmup")


def _wrap(middle: list[dict], rng: random.Random) -> list[dict]:
    """Discard the cold slot, pin a sentinel either side, randomise the interior."""
    body = list(middle)
    rng.shuffle(body)
    slots = [_warmup(), _sentinel()] + body + [_sentinel()]
    for i, s in enumerate(slots, start=1):
        s["position"] = i
    return slots


def noise_blocks(n_alloc: int, slots_per_alloc: int, seed: int = 0,
                 first_replicate: int = 1) -> list[dict]:
    """Stage 0. Every slot is `ref`; the only thing that varies is where it sat.

    No warm-up slot here, deliberately: quantifying the cold-start step is exactly
    what this stage is for, and it is what justifies discarding slot 1 everywhere else.
    """
    blocks = []
    for rep in range(first_replicate, first_replicate + n_alloc):
        slots = [_sentinel() for _ in range(slots_per_alloc)]
        for i, s in enumerate(slots, start=1):
            s["position"] = i
        blocks.append({"stage": "0", "replicate": rep, "block": 1,
                       "seed": seed, "slots": slots})
    return blocks


def screen_blocks(replicate: int, seed: int = 0) -> list[dict]:
    """Stage 2. The 20 PB runs split into two allocations on a dummy column.

    Blocking on a dummy keeps every real factor balanced 5-high/5-low within each
    allocation, so the allocation effect aliases with a dummy and never with a
    factor. The dummy used is the first column past the real factors.
    """
    design = pb20()
    block_col = len(FACTORS)
    if block_col >= len(design[0]):
        raise ValueError("no dummy column left to block on; PB-20 holds 19 factors max")

    blocks = []
    for sign, block_id in ((+1, 1), (-1, 2)):
        rng = _rng(f"2b{block_id}", replicate, seed)
        body = []
        for run_index, row in enumerate(design):
            if row[block_col] != sign:
                continue
            env, flags, levels = {}, BIND_FLAG, {}
            for col, factor in enumerate(FACTORS):
                high = row[col] == 1
                env.update(factor.high if high else factor.low)
                flags = factor.high_flags if high else factor.low_flags
                levels[factor.key] = 1 if high else -1
            body.append(_slot(f"pb{run_index:02d}", env, flags, "design",
                              levels=levels, run_index=run_index))
        body.append(_sham(f"b{block_id}"))
        # Without this the screen has no way to notice that no environment reached
        # the ranks at all: every factor would come back null and read as "the
        # defaults are already good", which is both a plausible result and, in that
        # scenario, completely wrong.
        body.append(_cap4())
        blocks.append({"stage": "2", "replicate": replicate, "block": block_id,
                       "seed": seed, "block_column": block_col,
                       "slots": _wrap(body, rng)})
    return blocks


def _grid_block(stage: str, replicate: int, seed: int, body: list[dict]) -> dict:
    return {"stage": stage, "replicate": replicate, "block": 1, "seed": seed,
            "slots": _wrap(body + [_cap4()], _rng(stage, replicate, seed))}


def algo_proto_block(replicate: int, seed: int = 0) -> dict:
    """Stage 3a. Full 3x3, because algo and proto interact strongly and because
    forcing either one globally overrides RCCL's per-size tuner -- the question is
    only whether the tuner picks wrong at the decode size."""
    body = []
    for algo in ("auto", "Ring", "Tree"):
        for proto in ("auto", "Simple", "LL128"):
            env = {}
            if algo != "auto":
                env["NCCL_ALGO"] = algo
            if proto != "auto":
                env["NCCL_PROTO"] = proto
            body.append(_slot(f"ap_{algo}_{proto}".lower(), env, BIND_FLAG, "design",
                              levels={"algo": algo, "proto": proto}))
    return _grid_block("3a", replicate, seed, body)


def min_nchannels_block(replicate: int, seed: int = 0) -> dict:
    """Stage 3c. The level ladder for the screen's one winner.

    Stage 2 measured NCCL_MIN_NCHANNELS=32 at +7.87% on the promotion scalar, but 32
    was a single arbitrary point. RCCL's own default here is 16 coll channels
    (job 22119061), so 32 is simply "twice the default" and nothing yet says whether
    the gain saturates, keeps climbing, or turns over.

    `auto` is the reference level and must stay in the ladder: without it the ladder
    measures differences between forced values and never says whether forcing helps
    at all.
    """
    body = []
    for level in ("auto", "24", "32", "48", "64", "96"):
        env = {} if level == "auto" else {"NCCL_MIN_NCHANNELS": level}
        body.append(_slot(f"minch_{level}", env, BIND_FLAG, "design",
                          levels={"min_nchannels": level}))
    return _grid_block("3c", replicate, seed, body)


def channels_buffsize_block(replicate: int, seed: int = 0) -> dict:
    """Stage 3b. The one interaction PB deliberately does not resolve."""
    body = []
    for cap in ("auto", "16", "8"):
        for buf in ("4194304", "8388608", "16777216"):
            env = {"NCCL_BUFFSIZE": buf}
            if cap != "auto":
                env["NCCL_MAX_NCHANNELS"] = cap
            body.append(_slot(f"cb_{cap}_{int(buf) // 1048576}m", env, BIND_FLAG, "design",
                              levels={"max_nchannels": cap, "buffsize": buf}))
    return _grid_block("3b", replicate, seed, body)


def verify_blocks(replicate: int, seed: int = 0) -> list[dict]:
    """Stage 1. One slot per knob, at the level that actually sets something.

    This stage measures nothing. It exists so that `verify_effect.py` has an
    NCCL_DEBUG=INFO capture per knob to assert against, and so a knob that is never
    read gets dropped here -- for the price of one 2-node job -- rather than after
    six allocations of screening have produced a null that means nothing.

    Low levels are mostly "unset", which is what the sentinel already is, so only the
    high level needs its own slot. `mr_monitor` is the exception: both of its levels
    set something, and kdreg2 is the one that has never been exercised here.
    """
    body = []
    for factor in FACTORS:
        if not factor.high and factor.high_flags == factor.low_flags:
            continue  # nothing to assert: the level changes no environment and no flag
        body.append(_slot(f"chk_{factor.key}", factor.high, factor.high_flags, "design"))
    # Not screened, but used by the Stage 3a grid, so they need the same proof.
    body.append(_slot("chk_algo", {"NCCL_ALGO": "Ring"}, BIND_FLAG, "design"))
    body.append(_slot("chk_proto", {"NCCL_PROTO": "LL128"}, BIND_FLAG, "design"))
    # The known-sign control has to be shown to take effect before it can be trusted
    # to prove that everything else does.
    body.append(_cap4())
    return [{"stage": "1", "replicate": replicate, "block": 1, "seed": seed,
             "slots": _wrap(body, _rng("1", replicate, seed))}]


def confirm_block(candidates: list[dict], replicate: int, seed: int = 0) -> dict:
    """Stage 4. Re-estimates the screen's winners on fresh allocations, because the
    top screened effect is biased upward by selection. env_tuned.sh quotes these
    numbers, never the Stage-2 ones."""
    body = [_slot(c["name"], c["env"], c.get("srun_flags", BIND_FLAG), "design")
            for c in candidates]
    return _grid_block("4", replicate, seed, body)


MANAGED_VARS = _all_managed_vars()

STAGES = {
    "0": lambda rep, seed: noise_blocks(1, 10, seed, first_replicate=rep),
    "1": verify_blocks,
    "2": screen_blocks,
    "3a": lambda rep, seed: [algo_proto_block(rep, seed)],
    # 3b's premise was a channels x buffsize interaction. Stage 2 measured buffsize at
    # -0.34% (p=0.76) and max_nchannels at +0.76% (p=0.51), so there is no pair of
    # effects left to interact. Kept runnable, but 3c is what the screen actually
    # pointed at.
    "3b": lambda rep, seed: [channels_buffsize_block(rep, seed)],
    "3c": lambda rep, seed: [min_nchannels_block(rep, seed)],
}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Print a stage's design as JSON.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, help="print only this block")
    args = parser.parse_args()

    blocks = STAGES[args.stage](args.replicate, args.seed)
    if args.block is not None:
        blocks = [b for b in blocks if b["block"] == args.block]
        if not blocks:
            raise SystemExit(f"stage {args.stage} has no block {args.block}")
    print(json.dumps(blocks if args.block is None else blocks[0],
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
