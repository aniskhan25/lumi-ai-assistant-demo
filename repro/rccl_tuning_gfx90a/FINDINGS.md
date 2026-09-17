# Findings — RCCL tuning on LUMI (gfx90a)

Every field below must be filled from `results/job_<jobid>/` output. **Nothing here is
a measurement until the runs have happened.** Every claim carries the Slurm job id that
produced it; superseded conclusions stay in place, struck through, rather than being
deleted.

Two rules this study adds to the ones inherited from `../rccl_startup_gfx90a/FINDINGS.md`:

1. **Pre-registration precedes submission.** `submit_stage.sh` prints a block; that
   block is committed here *before* the first `sbatch`. After that no threshold moves in
   light of the data.
2. **A null result on an unverified knob is not a null result.** If `knob_verified` is
   false, the row says `UNVERIFIED` and carries no number. Enforced in `analyze.py`, not
   just asserted here.

## Status

Nothing has run. The harness is built and its off-cluster checks pass:

| check | what it proves |
| --- | --- |
| `test_designs.py` | PB-20 is balanced 10/10 in every column and orthogonal across all 171 column pairs; blocking on a dummy keeps every factor 5/5 within each allocation; slot order is reproducible from `(stage, replicate, seed)`; `NCCL_NET_GDR_LEVEL` appears nowhere |
| `test_stats.py` | a null comparison rejects at ~5%; an injected 8% effect is recovered at 8%; the variance decomposition recovers injected 6%/2% components; Lenth's PSE resists one large real effect |
| `test_analyze.py` | **comparing the reference against itself across allocations produces no winner**; an injected +8% is promoted and a −6% killed; a knob whose environment never reached the ranks is reported `UNVERIFIED` and excluded |

Two harness bugs were found by those tests before any GPU time was spent, both of which
would have silently corrupted the study:

- The sentinel runs twice per block, first slot and last. Keyed on the variant name
  alone, the closing sentinel's per-rank JSON **overwrote the opening one**, so the drift
  gate had nothing to compare and could never fire. Fixed by putting the slot position in
  the filename and in the analysis key.
- `analyze.py` computed paired gains from a slot list that had already had the sentinels
  filtered out, so it had no reference to pair against and **silently emitted an empty
  gains table**. Fixed by filtering the output rather than the input.

## Pre-registrations

Paste each `submit_stage.sh` block here before submitting, then record the array ids.

<!-- ## Stage 0 pre-registration — paste from submit_stage.sh, then commit -->

## Hypotheses

Every row must resolve to confirmed, refuted, or underpowered. "Not tested" is not a
resolution; it moves the row to a stated limitation.

| # | hypothesis | why it is worth a design column | status | job ids |
| --- | --- | --- | --- | --- |
| T-1 | The position effect within an allocation is under 1%/slot, so variants may share one | decides whether the study costs ~260 or ~490 GPU-h | pending | |
| T-2 | `HSA_NO_SCRATCH_RECLAIM=1` cuts B0/B1 latency materially on gfx90a | AMD reports 5–10×; absent from every LUMI and HPE document | pending | |
| T-3 | `kdreg2` costs less than `userfaultfd` in registration-heavy traffic | both fix the hang; `env_baseline.sh:38` flags the comparison as never made | pending | |
| T-4 | MSCCL/MSCCL++ is not compiled into this container for gfx90a | would remove a factor before it costs allocations | pending | |
| T-5 | HPE's rendezvous set costs small-message latency | `RDZV_THRESHOLD=0` forces every message through rendezvous; measured inert for hangs, never for latency | pending | |
| T-6 | Raising `NCCL_MIN_NCHANNELS` helps where capping hurt | the bug study only ever capped | pending | |
| T-7 | No knob clears +3% on the promotion scalar — the defaults plus the monitor are already good | the likeliest outcome, and a shippable one | pending | |
| T-8 | A microbenchmark gain predicts the tok/s gain within [0.3×, 1.2×] | if it does not, the microbenchmark is measuring the wrong thing | pending | |

## Verdicts

| job id | stage | nodes | purpose | result |
| --- | --- | --- | --- | --- |
| | | | | |

## Controls

Every allocation must be able to answer all three. A failure voids the allocation or the
stage; it is recorded here either way.

| job id | sentinel drift | sham | `cap4` | env reached the ranks | kept? |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Limitations, stated up front

- **Scale.** Stages 2–3 run at 4 nodes. Anything found there is claimed for 4 nodes until
  Stage 4's 2/8-node tiers say otherwise.
- **Placement.** Allocations are randomised across Dragonfly groups, not blocked. The
  design detects a placement effect post hoc; it does not control it.
- **CPU-bind.** Measured in the guide's 8-task/node layout. The vLLM serving path runs
  1 task/node, where the mask is moot, so factor 12 is microbenchmark-only.
- **The promotion scalar is an upper bound on the achievable gain.** A tight loop entered
  from a barrier has perfectly synchronised ranks, nothing to overlap with and no HBM
  contention, so it understates in-serving collective cost.
- **`no-proxy` knobs.** `HSA_NO_SCRATCH_RECLAIM` and `NCCL_IGNORE_CPU_AFFINITY` cannot be
  shown from userspace to have taken effect. Their rows say so.

## Already reported upstream

| where | what | status |
| --- | --- | --- |
| `COMMENT-shs-ccl-docs.md` | the HPE guide's LUMI-specific corrections | draft, not sent |
| `COMMENT-laifs-tuning.md` | the measured LUMI baseline, for the container-recipes tracker | draft, not sent |
