# RCCL settings for LUMI (MI250X / gfx90a, Slingshot 11)

Measured, not inherited. Every line below resolves to a Slurm job id in `FINDINGS.md`.

**Read this first:** the useful answer is short, and most of it is negative. Of fourteen
variables HPE's tuning guide recommends, **one** matters on LUMI. Of ten further knobs
screened here, **none** is safe to recommend. That is the finding, not a failure to find
one — a list of settings that measurably do nothing on this machine is worth more than a
config full of variables nobody has tested.

Pinned stack: container `lumi-multitorch-u24r70f21m50t210-20260807_115122`, RCCL
`2.26.6-HEAD:64f48b6`, aws-ofi-nccl `1.20.0`, libfabric plugin v10, ROCr 1.18.
Findings below are for that stack; a different ROCm may behave differently and at least
one result (`HSA_NO_SCRATCH_RECLAIM`) probably does.

---

## 1. Set this. Every multi-node RCCL workload.

```sh
export FI_MR_CACHE_MONITOR=userfaultfd
```

Prevents an intermittent multi-node startup hang: 0 stalls in 18 attempts against a
baseline that hung 21 of 23 (jobs 21838111, 21838977, 21838978). Costs nothing
measurable — 88.1 vs 87.6 GB/s uncapped all-reduce (jobs 21838863, 21791400).

libfabric otherwise defaults to `memhooks` here, which detects remapping by intercepting
userspace allocator calls and does not reliably see ROCm memory operations, so a stale
registration is never invalidated and the transfer silently never completes
(job 21844179).

This is a **workaround** for an open bug (`laifs-container-recipes#44`), not a fix. Keep
generous startup timeouts.

Also worth setting, though not a comms variable — a persistent MIOpen cache saves ~167 s
of every cold multi-node launch (jobs 21822747, 21822748):

```sh
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
```

---

## 2. Never set these.

| variable | why | evidence |
| --- | --- | --- |
| `NCCL_NET_GDR_LEVEL=PHB` | **Unnecessary first, risky second.** ROCm ≥ 6.2 already defaults to this behaviour, and `laifs-container-recipes#30` records that "performance is still good without forcing the GDR level" — so there is nothing to gain. Separately it hung the first cross-node collective 32/32 at 4 nodes. Comes from HPE's `ccl_env.sh`. | 21790392 (4-node). **The 8-node figure previously quoted here came from job 21790393, whose table is retracted as a harness artefact — removed.** |
| `NCCL_MIN_NCHANNELS=32` **for vLLM with PP>1** | Server starts, reports `READY`, then **every request fails**. 0 of 120 requests served, 3 of 3 runs, against 3 of 3 reference runs serving 120/120 at all five concurrencies. Dies with a gloo timeout on the pipeline-parallel path at the first inference step. | 22143215, 22143216, 22143219 |
| `FI_MR_CACHE_MONITOR=kdreg2` | libfabric answers `kdreg2 monitor not available` and falls back to something that is neither kdreg2 nor userfaultfd. `/dev/kdreg2` exists on the nodes, so the device being present is not evidence the monitor works. | 22119061 |
| `NCCL_PROTO=LL128` | No LL128 path for bf16 all-reduce; the collective fails outright. **But this only bites if you force it** — see the note below. Not in HPE's guide, not recommended anywhere; it is a trap for people tuning, not a risk to normal users. | 22118353 |
| `NCCL_MIN_NCHANNELS` > 32 | **Silently discarded, not clamped.** RCCL's max is 32; a larger request is ignored and you get the default 16. Anyone bisecting upward sees the gain appear then vanish and concludes there is an interior optimum. There is not. | 22121788 |

### On `NCCL_PROTO`: the default already avoids this

RCCL's own tuner cost table on this stack (job 22119061, reference slot, nothing forced):

```
Algorithm  |            Tree            |            Ring            |
Protocol   |   LL  | LL128  | Simple    |   LL  | LL128  | Simple    |
AllReduce  |  0.0  |  0.0   |  0.0      | 21.5  | 12.0/0 |  36.8     |
```

Tree is all zeros — unavailable here — so **AllReduce runs on Ring**, and the tuner picks
`LL` or `Simple` by message size. The `LL128` column carries a zero on the inter-node
path, meaning the tuner already knows it is unusable and **never selects it**. So the
default (`NCCL_PROTO` unset) is correct and safe; the failure is reachable only by
forcing the protocol by hand.

---

## 3. Measured inert. Do not bother.

Each was screened at 4 nodes across 6 allocations, 40 high / 40 low runs per factor,
paired against a sentinel in the same allocation. Effects on the decode-weighted scalar
and on bulk bandwidth, both under 1.2% except where noted:

| variable | decode | bulk | note |
| --- | --- | --- | --- |
| `HSA_NO_SCRATCH_RECLAIM=1` | -0.12% | +0.40% | AMD documents a 5–10x small-message penalty on MI200 without it. **Not reproduced here.** Likely because AMD documents it for ROCm ≥ 7.13 and this container is 7.0 — but on the stack LUMI ships, it does nothing. |
| HPE rendezvous set (`FI_CXI_RDZV_PROTO/_THRESHOLD/_EAGER_SIZE/_GET_MIN`, `FI_CXI_DEFAULT_TX_SIZE`) | +2.15% | -0.59% | Presented as a recommendation; neither helps nor hurts at the sizes measured. `RDZV_PROTO=alt_read` additionally needs driver `rdzv_get_en=0` and falls back silently without it. |
| `NCCL_CROSS_NIC=1` | -1.00% | +0.01% | HPE recommends it "on large systems". No effect at 2–8 nodes here. |
| `NCCL_NET_GDR_READ=1` | -0.75% | +0.22% | |
| `NCCL_BUFFSIZE=8M` | -0.34% | -0.81% | |
| `NCCL_NCHANNELS_PER_NET_PEER=2` | +0.50% | -0.20% | |
| `NCCL_IGNORE_CPU_AFFINITY=1` | -0.99% | +0.00% | |
| `FI_CXI_DEFAULT_CQ_SIZE`, `FI_CXI_RX_MATCH_MODE` | — | — | Dropped before screening: two independent prior nulls. |
| `RCCL_MSCCLPP_ENABLE=1` | — | — | **Cannot be enabled.** RCCL: `MSCCL++: Cannot enable MSCCL++; environment is not MSCCL compatible` (22119061). |

**The CPU-bind mask is not in this table by accident.** Dropping the guide's 8-mask
layout measured -0.84% on bulk, i.e. nothing — but that was with 8 tasks/node. The vLLM
serving path runs 1 task/node where the mask is moot, so this result does not transfer to
a differently shaped job. Keep using the guide's masks; there is no evidence against them
and this study cannot speak to layouts it did not run.

---

## 4. The one knob that does something, and why you still should not set it

`NCCL_MIN_NCHANNELS=32` is the only variable out of ten that moved anything:

| scale | decode-weighted | bulk band |
| --- | --- | --- |
| 2 nodes | +15.45% | — |
| 4 nodes | **+7.87%** screened, **+7.55%** reproduced by an independent ladder | **+4.29%** |
| 8 nodes | **+0.50% ± 0.87** — nothing | — |

RCCL's default is 16 coll channels at 2, 4 and 8 nodes, and **32 is the maximum RCCL
allows** — `NCCL_MIN_NCHANNELS set by environment is ignored due to greater than max
allowed 32 channels` (job 22121788). So 32 is not a tuned value; it is the ceiling, and
the ladder was really `auto(16) -> 24 -> 32(max)`, with 48/64/96 being the default in
disguise. Note the channel knobs are **not in HPE's guide at all** — this candidate came
from AMD's RCCL usage tips, not from LUMI or HPE guidance.
It passed the 8-communicator hang gate cleanly (0 stalls, 160 ranks, 5 allocations,
job 22122188).

**And it breaks vLLM inference.** See §2. The microbenchmark evidence was strong,
consistent and reproduced across independent designs, and it was worthless: a collective
7.6% faster in a tight loop is not a faster server if the server cannot answer a request.

**24 fails the same way** (jobs 22236525-22236528, 0/120 requests in 3 of 3 runs), so
this is not about hitting RCCL's ceiling. At 24 the error is a deterministic
`Failed to CUDA host alloc 4923392 bytes`, which points at the mechanism: RCCL allocates
buffers per channel per peer, so raising the channel floor multiplies its pinned
host-memory footprint, and vLLM's 8-rank intra-node TP group plus cross-node PP pays that
cost many times over against a host already holding model weights.

**The cost of this knob scales with communicator count; the benefit does not.** That is
why every microbenchmark said yes: a single flat-world communicator allocates a fraction
of those buffers. Treat any channel-count result from a one-communicator benchmark with
suspicion.

For a **flat-world data-parallel** job — DDP or FSDP, one communicator, no pipeline stage
— the mechanism above suggests the cost may genuinely be lower, and +4.3% on
gradient-sized all-reduce at 4 nodes is real. **This study did not test that.** If you
want it, measure your own workload end to end, not just its collectives.

---

## 5. How to check a knob actually took effect

Most of the traps above are silent. A variable that is ignored produces a clean null
indistinguishable from a real one.

```sh
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING NCCL_DEBUG_FILE=/some/persistent/path/nccl_rank$SLURM_PROCID.log
grep -o '[0-9]* coll channels' nccl_rank0.log     # the REALISED count, not the requested one
```

- `NCCL_DEBUG_FILE` substitutes **`%h` and `%p` only** — there is no `%d` for rank. A
  `%d` is written literally and the file is never found.
- Point it at a **persistent** path. Node-local `/tmp` disappears with the allocation.
- INFO logging from every rank to Lustre is expensive enough to change what you are
  measuring — it took one 18-slot job from ~17 s a slot to 263 s. Use it to verify, never
  to measure.
- `Algorithm` and `Protocol` lines in the trace are the tuner's **cost-table header**,
  printed whether or not a knob is chosen. They prove nothing about what was selected.

---

## 6. What this study did not establish

- **Beyond 8 nodes.** Everything here is 2–8 nodes. The one real effect had already
  vanished by 8, and why is not explained.
- **Training end to end.** The bulk-band numbers are collective microbenchmarks. No
  training job was run.
- **Whether the vLLM failure is specific to PP>1**, to this vLLM version, or to the
  interaction of 32 channels with gloo's PP coordination. The serving config was fixed at
  `TP=8 PP=4` throughout, so these cannot be separated.
- **Contended conditions.** The noise floor was measured on a quiet fabric
  (allocation-to-allocation CV 0.09–0.30%). A busier machine would widen it.
