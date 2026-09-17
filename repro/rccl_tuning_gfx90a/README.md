# A measured RCCL tuning baseline for LUMI (gfx90a)

HPE's [`rccl_tuning_guide.md`](https://github.com/HewlettPackard/shs-ccl-docs/blob/main/rccl/rccl_tuning_guide.md)
is the only thing resembling an RCCL tuning reference for Slingshot machines, and it
is the wrong instrument for LUMI. It is not stale — its last commit is 2026-08-25 —
it is the **wrong genre**: a generic correctness/config list of 14 variables, with no
benchmarking methodology, no gfx90a specifics, and not one performance knob. No
channels, no algorithm, no protocol, no buffer size, no MSCCL.

The sibling investigation in `../rccl_startup_gfx90a/` already measured, on LUMI:

- 10 of its 11 RCCL-list variables are **inert** — the full HPE set gives 0/5 stalls,
  the same set minus `FI_MR_CACHE_MONITOR` gives 4/5, identical to baseline (job 21838977);
- `NCCL_NET_GDR_LEVEL=PHB` **deterministically hangs** the first cross-node collective
  at 4 and 8 nodes (jobs 21790392, 21790393, 21794114);
- `NCCL_SOCKET_IFNAME=hsn0..3` is **refuted** — every apparent win was cache warming
  or position in the job (jobs 21818930/1, 21822747/8).

Meanwhile the guide omits everything AMD now documents for MI250X: `HSA_NO_SCRATCH_RECLAIM=1`
(AMD reports a 5–10× small-message latency penalty on gfx90a without it),
`RCCL_MSCCL_FORCE_ENABLE=1` (MSCCL is off by default on non-MI300X), and the whole
channel/buffsize/algo/proto axis.

So there is no measured RCCL **performance** baseline for LUMI. The bug study stopped
exactly where one becomes possible: it established a correct starting point and left a
list of explicitly unmeasured candidates in `../rccl_startup_gfx90a/env_baseline.sh:71-96`,
including `kdreg2` vs `userfaultfd`, which line 38 flags as never compared.

**What this ships:** `env_tuned.sh`, in which every line carries a Slurm job id and a
measured cost, plus upstream reports correcting the guide for LUMI. Primary target is
vLLM inference serving; the training band is measured and reported too, but knobs are
promoted on the inference band.

## Pinned environment

| | |
| --- | --- |
| Container | `/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-...sif` (pinned, never the `latest` symlink) |
| Account | `project_462000131` |
| RCCL | `librccl.so.1.0.70002`, net plugin `/usr/lib/x86_64-linux-gnu/librccl-net-ofi.so` |
| libfabric | 1.22.0, CXI provider |
| Task layout | the official guide's lesson 05: 8 tasks/node, 7 cpus/task, `--mem-per-gpu=60G`, `--cpu-bind=v,mask_cpu=<8 masks>` |
| Partitions | `dev-g` for stages 0–1, `standard-g` for 2 onward, **never mixed within a stage** |
| Serving reference | Kimi-K2-Instruct-0905, `TP=16 PP=2`, expert parallel, `--max-num-seqs 64`, 3.648 tok/s per GCD |

Override with `STAGE=`, `REPLICATE=`, `BLOCK=`, `SEED=`, `CONTAINER=`, `PROFILE_ARGS=`,
`SLOT_DEBUG=`. Results land in `results/job_<jobid>/` (gitignored).

## The one constraint everything is built around

`../rccl_startup_gfx90a/FINDINGS.md:470` records that repeats within one allocation are
not independent samples, and that five false findings came from ignoring it. A tuning
study is far more exposed than a hang study, because it reads *differences in timings*
rather than binary stall/no-stall.

| noise term | source | how it is handled |
| --- | --- | --- |
| σ_within | rep to rep inside one `srun` | averaged; raw samples kept so it can be estimated |
| σ_pos | slot to slot inside an allocation | randomised order, position as a covariate, cold slot discarded |
| σ_alloc | allocation to allocation: Dragonfly span, neighbours | **cancelled by pairing against a sentinel in the same allocation** |

One allocation is one block:

```
[ warmup ] [ S ] [ V1 V2 … Vn in random order ] [ S' ]
```

`warmup` is measured and thrown away — the first slot pays MIOpen compilation, Lustre
first touch and a cold MR cache, so without it `log(S'/S)` would report cold start
rather than drift and the gate would discard every block. `S` and `S'` are the same
reference; their ratio is the drift estimate. Every variant's number is a **paired
contrast against the sentinel mean of its own allocation**, so σ_alloc never enters.

Node placement is recorded and stratified, never pinned with `--nodelist`: pinning
would tune for one node set and give no evidence the answer generalises across
Dragonfly groups, which is the dimension most likely to matter for `NCCL_CROSS_NIC`
and the CXI knobs.

## Run it, cheapest first

**0. The noise floor. This is the budget gate — nothing after it is committed.**

```bash
./submit_stage.sh 0             # prints the pre-registration, submits nothing
CONFIRM=1 ./submit_stage.sh 0   # after committing that text to FINDINGS.md
```

10 allocations, ~85 GPU-h, one variant (`ref`) in every slot. Produces σ_alloc, σ_pos,
the position slope fitted on slots 2+, the slot-1 step estimated separately, and the
MDE ladder. **Stop here and re-decide the budget with a real MDE in hand.**

**1. Does each knob take effect?** ~21 GPU-h. Cheaper than discovering after six
allocations of screening that a knob was never read.

```bash
CONFIRM=1 ./submit_stage.sh 1
python3 verify_effect.py --results-dir results/job_<id>
```

**2. The screen.** PB-20 over 12 factors, 3 replicates × 2 blocks, ~63 GPU-h.

```bash
CONFIRM=1 ./submit_stage.sh 2
python3 analyze.py --results-root results --cv-pos-pct <from stage 0>
```

**3. The two interactions the screen deliberately does not resolve.**

```bash
CONFIRM=1 ./submit_stage.sh 3a   # NCCL_ALGO x NCCL_PROTO
CONFIRM=1 ./submit_stage.sh 3b   # NCCL_MAX_NCHANNELS x NCCL_BUFFSIZE
```

**4. Confirmation on fresh allocations, then the hang regression gate.** The screen
nominates; Stage 4 prices. `env_tuned.sh` quotes Stage-4 numbers, never Stage-2 ones.
Then the candidate must pass `../rccl_startup_gfx90a/run_rccl_probe.sh` at 4 nodes,
5 attempts, **0/5 hangs** — a tuning config that reintroduces the memhooks-class hang
is worthless at any tok/s.

**5. End to end.** `../../run_vllm_bench_multinode.sh` unmodified, paired AB/BA
crossover, one arm restart per arm. Promote only if all four hold: sign consistent in
3 of 3 pairs and in both orders; mean paired gain ≥ +2% tok/s per GCD; the observed
gain within [0.3×, 1.2×] of what the promotion scalar predicted; no regression in
`latency_p95_s`, failures or startup.

## Bands, and why these ones

`../rccl_startup_gfx90a/summarize_bandwidth.py` headlines 128 MiB – 1 GiB. For the
shipped serving config that is ~150× larger than anything this repo actually sends:

```
S_decode  = hidden x max_num_seqs x 2 B       = 7168 x 64 x 2         ~ 896 KiB
S_ep_a2a  = tokens x topk x hidden x 2 / EP   = 64 x 8 x 7168 x 2/16  ~ 448 KiB
S_prefill = 3200 batched tokens x 7168 x 2                            ~  44 MiB
```

| band | range | reported as | why |
| --- | --- | --- | --- |
| B0 floor | 8 B – 8 KiB | median + p95 µs | fixed fabric and RCCL cost |
| **B1 decode** | 16 KiB – 1 MiB | median + p95 µs | **the band that decides this study** |
| B2 prefill | 4 MiB – 128 MiB | bus bandwidth GB/s | chunked prefill |
| B3 bulk | 256 MiB – 2 GiB | bus bandwidth GB/s | continuity with job 21791400; not in the scalar |

1–4 MiB is measured but in no band: RCCL changes protocol somewhere in there, and a
bandwidth averaged across a regime change describes neither regime. Small messages are
reported as latency, never bandwidth — "0.02 GB/s" at 8 B is unusable, "31 µs" is not.

The promotion scalar is **predicted collective milliseconds per decode step**,
`61 × t_allreduce(896 KiB) + 60 × t_alltoall(448 KiB)`. It is a prediction, not a
summary: with the collective fraction of a decode step it says how much end-to-end gain
to expect, which is what makes Stage 5 a test rather than a hope. It is a lower bound on
in-serving cost and so an **upper** bound on the achievable gain — a tight loop entered
from a barrier has perfectly synchronised ranks, nothing to overlap with and no HBM
contention.

## What each file does

| file | role |
| --- | --- |
| `designs.py` | the whole design as a pure function of (stage, replicate, seed): PB-20, the grids, the per-block permutation. Pre-registerable and reproducible |
| `run_block.sh` | one allocation = one block. Unsets every managed variable between slots, fresh `MASTER_PORT`, reaps stragglers, records `slot_status.tsv` |
| `collective_profile.py` | the measurement. Descends from `bandwidth_sweep.py` |
| `analyze.py` | paired contrasts, bands, controls, variance decomposition, PB effects → `tuning.md` |
| `stats.py` | the statistics, standard library only, so laptop and container agree |
| `verify_effect.py` | Stage 1. Emits `knob_verified.json`/`.md` |
| `submit_stage.sh` | prints the pre-registration, then submits the stage as a job array |
| `env_tuned.sh` | **the deliverable.** A template until jobs justify each line |
| `KNOBS.md` | the catalogue: per knob, source, prior, hypothesis, how it could silently do nothing |
| `FINDINGS.md` | the lab notebook. Every claim carries a job id |
| `test_*.py` | off-cluster checks. Run all three before submitting anything |

## Controls, and how to read a verdict

| control | rule |
| --- | --- |
| **drift sentinel** `S`, `S'` | `\|log(S'/S)\| > 3σ_pos` → discard the allocation and resubmit |
| **sham** — `ref` under a different name | if it reaches 3%, every positive in that stage is suspect |
| **known-sign** `cap4` | must come back strongly negative, or the harness is not reaching the ranks — stop the stage |

| verdict | meaning |
| --- | --- |
| `promote` | ≥ +5%, interval excludes 0, consistent sign in ≥ 3 allocations |
| `hold` | +3% to +5%; retested only in combination at Stage 4 |
| `inert` | \|Δ\| < 3% with CI half-width < 3% — **ship it commented out, with the cost stated** |
| `underpowered` | \|Δ\| < 3% but the interval is too wide to say so |
| `kill` | ≤ −5%; into the `DO NOT SET` section with its job id |
| `UNVERIFIED` | the knob could not be shown to take effect. **Not a null result** |

The 3% floor is hard regardless of significance: across a dozen screened factors a 2%
"significant" result is mostly multiplicity, and a 2% microbenchmark gain could never
clear the Stage-5 acceptance rule, so it is unshippable either way.

**Expect most knobs to come back inert.** Given that the bug study found every apparent
win to be position or cache warming, "the defaults plus the monitor are already good" is
a likely and publishable conclusion. A list of knobs measurably worthless on LUMI gfx90a,
each with a job id, is worth more to the next person than a config full of cargo-culted
variables — which is exactly what `env_baseline.sh`'s own header argues.

## Before submitting anything

```bash
python3 test_designs.py && python3 test_stats.py && python3 test_analyze.py
```

The one that matters is in `test_analyze.py`: comparing the reference against itself
across allocations must produce no winner. A pipeline that reports a winner on null
input would have reported one on the real data too, and nobody could have told.

## Findings

Record conclusions in `FINDINGS.md`. Nothing there is a measurement until a job has
produced it, and every claim carries the Slurm job id that did.
