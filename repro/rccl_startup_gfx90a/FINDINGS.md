# Findings: multi-node vLLM startup stalls on LUMI (RCCL/Slingshot, gfx90a)

> Template. Every field below must be filled from `results/job_<jobid>/` output.
> Nothing here is a measurement until the runs have happened. Every claim carries the
> Slurm job id that produced it; superseded conclusions stay in place, struck through,
> rather than being deleted.

## Setup

| | |
| --- | --- |
| Container | `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif` (pinned) |
| RCCL | 2.26.6-HEAD:64f48b6 (file `librccl.so.1.0.70002`); net plugin `librccl-net-ofi.so`, in-container |
| libfabric | 2.1.0 in-container, 1.22.0 on the host; CXI provider |
| Partition | `dev-g` for rungs 1/2/6/7, `standard-g` for rungs 4/5 |
| Account | `project_462000131` |
| Node interfaces present | `hsn0..hsn3` (Slingshot), plus `nmn0`, `net1`, `net2`, `net3`, `bond0` |

## What was already true before any job ran

These are facts about the repo and the system, established by reading rather than by
measurement, and they frame everything below.

| # | Fact | Source |
| --- | --- | --- |
| F1 | This repo sets no `NCCL_*`/`RCCL_*`/`FI_CXI_*` variable anywhere, in the working tree or in any commit | `git log --all -S'NCCL_'`, `-S'FI_CXI'`, `-S'aws-ofi'` all empty |
| F2 | `lumi-aif-singularity-bindings` sets only `SINGULARITY_BIND` and `SLURM_MPI_TYPE=pmi2` — the bindings configure defaults, but none of them are comms tuning | `/appl/local/laifs/modules/lumi-aif-singularity-bindings/1.0.{0,1}.lua` |
| F3 | Our own multi-node recipes already need `STARTUP_TIMEOUT_S` of 2700 / 3600 / **14400** s and all reached health | root `README.md`, "Successful Launch Commands" |
| F4 | The elapsed startup time is computed and discarded, so those minutes were never attributed | `run_vllm_demo_multinode.sh:62` vs `:67` |
| F5 | The LUMI AI Guide sets `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` and `NCCL_NET_GDR_LEVEL=PHB` in lesson 5, but **not** in lesson 3 where multi-node RCCL lives | `5-experiment-tracking/run_*.sh` vs `3-multi-gpu-and-node/*` |
| F6 | The guide prescribes no `NCCL_MAX_NCHANNELS`, `NCCL_RUNTIME_CONNECT`, `FI_CXI_*` or `FI_PROVIDER` anywhere — so no measured comms baseline for LUMI exists yet | grep across the whole guide |
| F7 | `run_vllm_demo_multinode.sh` has no LAIFS reachability guard, so a stalled `/appl/local/laifs` mount is indistinguishable from the stall under investigation | absent vs `repro/mistral3_gfx90a/run_repro.sh:44-51` |
| F8 | The multi-node launcher sets `RUNAI_STREAMER_CONCURRENCY=1` / `MEMORY_LIMIT=8` but never passes `--load-format runai_streamer`, so they are inert there; the README's recipes pass concurrency 4 and also never pass the flag | `launch_vllm_multinode_rank.sh:27-28` vs `:30-41` |
| F9 | Weights are read from `/scratch` (Lustre) in every launcher; `/flash` (NVMe) is never used. `Kimi-K2-Instruct-0905` is 959 GB | `HF_HOME` in the launchers; `du` on the cache |
| F10 | MIOpen caches to a per-job `mktemp -d`, so kernel compilation is re-paid every launch; the guide instead pins a persistent per-user path | `launch_vllm_multinode_rank.sh:16-18` vs the guide's `setup.sh` |

## Rung 1: no stall at 2 nodes, and the cost is in the first collective (job 21790359)

2 nodes, world size 16, `dev-g`, 84 s wall clock for all three variants. Every one of
the 48 ranks reported `COMPLETED`; **nothing stalled**.

| variant | verdict | startup_s | init | world_first | world_second | fresh_world_first | tp_like_first | pp_like_first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nchannels_8` | FAST | 12.87 | 0.75 | 9.51 | 0.00 | 1.21 | 1.14 | 0.26 |
| `socket_ifname` | FAST | 13.49 | 0.78 | 9.82 | 0.00 | 1.41 | 1.25 | 0.24 |
| `baseline` | FAST | 18.34 | 0.76 | 14.71 | 0.00 | 1.43 | 1.22 | 0.22 |

What this establishes, and what it costs the two hypotheses:

- **The stall does not reproduce at 16 ranks.** 18 s is not 10-60 minutes. Whatever the
  reporter is hitting needs more ranks than this, so the 4- and 8-node runs are the
  real test, not this one.
- **`init` is 0.76-0.78 s in all three variants**, and pinning the interface did not
  move it (0.78 vs 0.76). So H-A as originally stated — a slow *bootstrap* over a
  management interface — contributes nothing measurable at this scale. That specific
  mechanism is **refuted at 2 nodes**.
- **But `socket_ifname` still cut `world_first` by 33%** (14.71 s → 9.82 s). Since it
  did not touch `init`, the benefit is not in the bootstrap at all: it is in which NICs
  RCCL builds its *data-path* connections over. H-A survives in a revised form —
  interface selection matters, just not where predicted.
- **`world_first` 14.71 s vs `world_second` 0.00 s** is the connection-setup signature:
  the first collective pays for establishing connections and the second pays nothing.
- **But `fresh_world_first` is only 1.43 s**, an order of magnitude below `world_first`.
  A second communicator over the same 16 peers is nearly free. So the strong form of
  H-B — every new communicator pays a comparable burst — is **not supported at this
  scale**. Most of the 14.71 s is one-time (topology discovery, plugin and channel
  setup), and the per-communicator increment is ~1.2-1.4 s. Whether that increment
  grows superlinearly with ranks is exactly what the 8-node run decides.
- **`nchannels_8` cut `world_first` by 35%** (14.71 s → 9.51 s), so channel count does
  drive first-collective setup cost even where there is no stall to fix.

### RCCL cannot find a network path for GCDs 1, 3 and 7

Every variant emitted exactly 64 instances of

```
NCCL WARN .../graph/topo.cc:1550 Could not find any local path from gpu N to net
```

for `N` in {1, 3, 7} — identical counts in `baseline`, `socket_ifname` and
`nchannels_8`, so it is a constant of the environment rather than something these
variants change. On a LUMI-G node with 4 Slingshot NICs and 8 GCDs, three GCDs having
no local path to the network means their traffic has to route via a peer GCD. This is
present at defaults, in the current pinned container, and is worth reporting on its own
account regardless of how the stall resolves.

### Environment, as measured

| | |
| --- | --- |
| RCCL | 2.26.6-HEAD:64f48b6, `/opt/rocm-7.0.2/lib/librccl.so.1` |
| HIP / ROCm | 7.0.51831-7c9236b16 / 7.0.2.0-56-9428210 |
| torch | 2.10.0+rocm7.0 |
| arch | `gfx90a:sramecc+:xnack-`, 8 devices visible per node |
| CXI devices | `cxi0..cxi3` |
| Net interfaces | `hsn0`, `hsn1`, `nmn0`, `net1`, `net2`, `net3`, `bond0`, `lo`, `can0` |
| OFI plugin | `/usr/lib/x86_64-linux-gnu/librccl-net-ofi.so`, 481 KB ELF (present and real) |
| libfabric | 2.1.0 inside the container (`/usr/bin/fi_info`); 1.22.0 on the host |

Two harness bugs this run exposed, both fixed: `iproute2` is not installed in the LAIF
container, so the interface list has to come from `/sys/class/net`; and
`/opt/venv/bin/fi_info` is a wrapper pointing at a non-existent path while shadowing a
working `/usr/bin/fi_info` on `PATH`.

## Rung 2: `NCCL_NET_GDR_LEVEL=PHB` hangs the first cross-node collective (jobs 21790392, 21790393)

Reproducible at two scales, all ranks, every time:

| nodes / world | variant | init | world_first | status |
| --- | --- | --- | --- | --- |
| 4 / 32 | `baseline` | 0.83 | 15.71 | ok |
| 4 / 32 | `socket_ifname` | 0.49 | **9.63** | ok |
| 4 / 32 | `gdr_level` | 0.70 | **300.00** | **STALL 32/32 in `world_first`** |
| 8 / 64 | `baseline` | 0.85 | 15.82 | ok |
| 8 / 64 | `socket_ifname` | 0.63 | **9.79** | ok |
| 8 / 64 | `gdr_level` | 1.42 | **300.00** | **STALL 64/64 in `world_first`** |

`NCCL_NET_GDR_LEVEL=PHB`, on its own, with nothing else changed, hangs the first
cross-node collective indefinitely — 300.00 s is the watchdog timeout, not a
measurement. `init` completes normally in the same runs, so the failure is in
connection setup for the data path, not in the rendezvous.

This is a setting **the LUMI AI Guide recommends**, in
`5-experiment-tracking/run_*.sh`. Those are single-node jobs, where it is harmless. It
is absent from `3-multi-gpu-and-node/`. The planning note for this case treated that
absence as a gap in the guide; it is the opposite — lesson 3 is right to omit it, and
copying it from lesson 5 into a multi-node job is actively harmful.

It plugs into finding 11: RCCL reports no local path to the network for GCDs 1, 3 and 7.
Forcing GPU-direct RDMA across the host bridge for GCDs that have no working path is a
coherent mechanism for a hang rather than a fallback. Not yet proven to be *the*
mechanism.

### Why this is probably not the reporter's stall

Checked before drawing any conclusion:

| Fact | Evidence |
| --- | --- |
| The container sets no `NCCL_*`/`RCCL_*`/`FI_*` variable at all | `singularity exec ... env \| grep -E '^(NCCL_\|RCCL_\|FI_\|OFI_)'` is empty; `/.singularity.d/env/` has none |
| Nothing under `/appl/local/laifs` sets `NCCL_NET_GDR_LEVEL` | `grep -rli` finds no file |

So nobody gets this setting by accident from the platform — it has to be set
deliberately. **The reporter never mentioned setting it.** Until they confirm whether
their job environment carries it, this is a hazard this investigation found, and a
guide bug, but *not* an established explanation of their stall. Worth asking them
directly, because if they do set it, it explains their report end to end.

### `socket_ifname` is a consistent, scale-independent win

`world_first` drops from 15.7-15.8 s to 9.6-9.8 s — about 38% — at 16, 32 and 64 ranks
alike, with `init` unchanged. Cheap, and independent of how the stall resolves.

### The reporter's stall did not reproduce at their own scale

`baseline` at 8 nodes / world 64 — the same geometry as the frozen job in the report —
completed in 19.97 s with zero stalled ranks. And the cost is flat in rank count:
`world_first` is 14.71 s at 16 ranks, 15.71 s at 32, 15.82 s at 64. So a pure RCCL
workload does not exhibit the reported failure, and connection setup does not blow up
with world size in the range that matters.

That points away from "RCCL on LUMI is broken at 64 ranks" and toward either an extra
setting in their environment (see above) or something specific to vLLM's startup —
many more communicators, concurrent creation, or 8 unbound workers per task. The
`many_comms` phase and the vLLM rung exist to test the latter.

## RETRACTION: most of the 8-node sweep was a harness artefact, not a finding

Job 21790393 (8 nodes) reported 11 of 13 variants stalling, in varied phases. That
table is **invalid** and must not be quoted. Two things give it away:

1. **The positive control failed.** `net_socket` stalled. `README.md` states the rule in
   advance: `net_socket` takes RCCL off the CXI path entirely, so if it stalls, the
   harness or the allocation is at fault rather than RCCL. That rule was written before
   the run and it applies to the run.
2. **58 of 64 ranks stalled in `world_second`** — a repeat all-reduce on an already-warm
   communicator, which completes in 0.00 s whenever it is healthy. There is no physical
   RCCL failure that hangs a warm collective while leaving `init` at 0.6 s. That is
   leaked state, not a fabric problem.

**Cause, found and fixed.** The probe passes `--kill-on-bad-exit=0` so that one stalled
rank does not abort the whole job. The consequence, which the harness did not handle,
is that when a variant stalls, the ranks that did *not* trip their watchdog stay alive
and blocked inside RCCL, still holding their GCDs, while the next variant starts. The
next variant then stalls for reasons that have nothing to do with its setting, and the
cascade reads as a sweep full of discoveries. `run_vllm_startup.sh` had the same bug
and it was fixed there; `run_rccl_probe.sh` was missed.

The fix reaps leftover probe processes on every node after every attempt, and adds a
**canary**: a plain 1 KiB all-reduce between variants that must stay healthy. When the
canary fails, the log says the node set is contaminated and later rows are not valid,
instead of printing numbers that look like measurements.

### What survives the retraction

Variant order is `baseline, socket_ifname, gdr_level, guide_pair, runtime_connect_off,
nchannels_8, nchannels_4, nchannels_16, nchannels_per_peer, cxi_cq_and_sw_match,
proto_simple, cpu_bind, net_socket`. Rows up to and including the **first** stall in a
job cannot have been contaminated, because nothing had stalled yet.

| Claim | Status |
| --- | --- |
| `baseline` healthy at 16, 32 and 64 ranks; `world_first` 14.71 / 15.71 / 15.82 s (flat in rank count) | **valid** — first variant in every job |
| `socket_ifname` healthy and ~38% faster at all three scales | **valid** — second variant, before any stall |
| `gdr_level` (`NCCL_NET_GDR_LEVEL=PHB`) stalls all ranks in `world_first` at 4 and 8 nodes | **valid** — third variant, the first stall in each job, so uncontaminated, and reproduced independently at two scales |
| `guide_pair` also stalls | **probable but not clean** — it ran immediately after `gdr_level` stalled. Consistent at both scales and it contains the same poison, but it needs a re-run to be stated as fact |
| everything from `runtime_connect_off` onward, at both scales | **retracted** — ran after a stall, contamination cannot be excluded |
| the whole 2-node rung 1 table (job 21790359) | **valid** — no variant stalled, so there was nothing to contaminate |

### A claim I made and now withdraw

On the strength of the contaminated rows I reported that intermittent
`fresh_world_first` stalls — `nchannels_16` at 4 nodes, `runtime_connect_off` at 8 —
were the reproduction of the reporter's "stalls at different phases" symptom. **They
are not.** Both ran after `gdr_level` and `guide_pair` had stalled on the same nodes,
so the varied phases are the signature of leaked state, and the varied-phase symptom is
exactly what contamination would counterfeit. The reporter's symptom remains
**unreproduced** in a clean pure-RCCL run.

## Q3 ANSWERED: capping channels at 8 costs ~20% of collective bandwidth; capping at 16 costs nothing (job 21791400)

8 nodes, world size 64, median of 20 reps with 5 discarded, bus bandwidth on the
nccl-tests convention. This is the reporter's own scale.

**Training band (128 MiB - 1 GiB), bus bandwidth GB/s** — where gradient all-reduce
buckets in bandwidth-bound training sit:

| variant | all_gather | all_reduce | reduce_scatter | bandwidth lost |
| --- | --- | --- | --- | --- |
| `default_channels` | 82.3 | 87.6 | 82.3 | reference |
| `NCCL_MAX_NCHANNELS=16` | 82.5 | 87.6 | 82.5 | **none measurable (-0%)** |
| `NCCL_MAX_NCHANNELS=8` | 74.1 | 70.5 | 77.3 | **20%** |
| `NCCL_MAX_NCHANNELS=4` | 41.9 | 38.7 | 41.3 | **56%** |

**Small messages (<= 1 MiB), bus bandwidth GB/s** — no measurable effect from any cap:

| variant | all_gather | all_reduce | reduce_scatter |
| --- | --- | --- | --- |
| `default_channels` | 0.22 | 0.69 | 0.23 |
| `NCCL_MAX_NCHANNELS=16` | 0.21 | 0.69 | 0.22 |
| `NCCL_MAX_NCHANNELS=8` | 0.22 | 0.70 | 0.22 |
| `NCCL_MAX_NCHANNELS=4` | 0.28 | 0.56 | 0.28 |

Three things follow, and the second is the useful one:

1. **The cost is entirely at large messages.** Small-message latency is untouched by any
   cap. So an inference server, whose collectives are small per decode step, would
   barely notice `NCCL_MAX_NCHANNELS=8`; a training job doing multi-hundred-MB gradient
   all-reduce pays the full 20%. The reporter's instinct that this is the wrong knob for
   bandwidth-bound training is correct.
2. **`NCCL_MAX_NCHANNELS=16` is free.** Within measurement noise it is identical to
   uncapped across all three collectives. If capping channels is what cures the stall,
   the obvious thing to try is 16 rather than 8 — it may buy the fix at no bandwidth
   cost at all. Whether 16 is *enough* to stop the stall is a separate question this
   job does not answer, and it is worth testing before recommending.
3. **The harness measures something real.** `README.md` set the sanity check in advance:
   uncapped large-message all-reduce must land in the plausible range for 4x200 Gb/s of
   Cassini per node, i.e. ~100 GB/s. Measured 87.6 GB/s. Close to the per-node NIC
   ceiling without exceeding it, so the numbers can be trusted.

## REPRODUCED: the reporter's stall, at defaults, in pure RCCL (job 21794113)

8 nodes, world size 64 — the reporter's own geometry. `baseline` means no `NCCL_*` or
`FI_CXI_*` variable set at all, which per F1/F2 is exactly what this repo and the
platform give you.

| variant | init | world_first | fresh_world_first | status |
| --- | --- | --- | --- | --- |
| `baseline` | 0.82 | 15.20 | **150.00** | **STALL 64/64 in `fresh_world_first`** |
| `socket_ifname` | 0.74 | 9.87 | 1.56 | ok |
| `runtime_connect_off` | 0.73 | 9.92 | 1.55 | ok |
| `nchannels_8` | 0.76 | 9.72 | 1.33 | ok |
| `nchannels_4` | 1.30 | 9.83 | 1.22 | ok |
| `nchannels_16` | 0.70 | 9.82 | 1.55 | ok |
| `nchannels_per_peer` | 0.72 | 9.88 | **150.00** | **STALL 64/64 in `fresh_world_first`** |
| `cxi_cq_and_sw_match` | 1.23 | 10.43 | 1.60 | ok |
| `proto_simple` | 5.03 | 14.22 | 1.56 | ok |
| `cpu_bind` | 0.76 | 9.89 | **150.00** | **STALL 64/64 in `fresh_world_first`** |
| `net_socket` | 0.73 | 9.62 | 1.25 | ok |

**Why this table is valid**, unlike the one retracted above:

- **The positive control passed.** `net_socket` completed in 9.62 s. That was the
  stated pass condition, written before the run.
- **`baseline` stalled as the first variant in the job**, so nothing preceded it that
  could have contaminated it.
- **Recovery after every stall.** `socket_ifname` was healthy immediately after
  baseline's 64-rank stall, and six more variants were healthy after it. Contamination
  can only *cause* stalls, never cure them, so repeated recovery is positive evidence
  that the reaping fix works.

**What is reproduced.** At 64 ranks, with no tuning, RCCL hangs indefinitely when a
**second communicator** is created and first used — `fresh_world_first` — having already
completed the first collective successfully in 15.20 s. This is the reporter's symptom
stated precisely: getting through one stage does not prevent a stall at a later one,
because each new communicator is a fresh opportunity to hang.

**It is intermittent.** The same `baseline` at the same 64 ranks was healthy in job
21790393 (1.56 s) and in job 21794114 (1.55 s), and stalled in 21794113. Three
observations, one stall. That is why the reporter sees different runs freeze at
different stages, and it is why a single run — in either direction — proves nothing.
Job 21811023 repeats `baseline`, `nchannels_16`, `nchannels_8` and `net_socket` five
times each to get an actual stall rate.

`nchannels_per_peer` and `cpu_bind` also stalled in the same phase. Given the
intermittency those may be nothing more than further samples of the same race, and must
not be read as effects of their settings without repeats.

## Q2 PARTIALLY ANSWERED: capping channels rescues an OFI/CXI setup hang; eager connect does not (job 21794114)

Combination rows, run after the GDR hang was established. These deliberately move more
than one variable, to ask which remedy rescues a *known* hang.

| variant | world_first | status |
| --- | --- | --- |
| `baseline` | 14.77 | ok |
| `gdr_level` (`NCCL_NET_GDR_LEVEL=PHB`) | **150.00** | **STALL 64/64** |
| `gdr_cap` (+ `NCCL_MAX_NCHANNELS=8`) | 9.71 | **ok — rescued** |
| `gdr_runtime_connect` (+ `NCCL_RUNTIME_CONNECT=0`) | **150.00** | **STALL 64/64** |
| `gdr_ifname` (+ `NCCL_SOCKET_IFNAME=hsn0-3`) | **150.00** | **STALL 64/64** |
| `gdr_socket_net` (+ `NCCL_NET=Socket`) | 9.52 | **ok — rescued** |

Three things this establishes:

1. **`NCCL_MAX_NCHANNELS=8` rescues a hang that nothing else does.** Eager connect and
   the interface pin both fail to. This is a mechanism for *why the reporter's
   workaround works* rather than a guess: fewer channels means fewer concurrent
   connection setups, and the hang is in connection setup.
2. **`NCCL_NET=Socket` also rescues it**, which places the hang in the OFI/CXI path
   specifically — consistent with the reporter's own observation that `NCCL_NET=Socket`
   avoids their stall, and with `FI_CXI_*` tuning doing nothing.
3. **`gdr_cap` was healthy immediately after `gdr_level` stalled 64/64** on the same
   nodes, which independently confirms the reaping fix.

## The canary is unreliable and is now off by default

The between-variant canary added after the retraction reported "node set contaminated"
after **every** variant in both jobs 21794113 and 21794114 — including after `baseline`
runs that had just completed healthily, and after the healthy `net_socket` control. A
check that fails on known-good runs is worse than no check: taken at face value it
would have discarded the valid table above. It is disabled by default
(`CANARY=1` re-enables, `CANARY_LOG=<path>` captures its own error output) and its
verdicts must not be used until diagnosed. Validity is instead argued from the positive
control and from recovery-after-stall, both of which come from the measurement itself.

## Stall rates, and why 5 attempts is not enough (job 21811023)

8 nodes, world size 64, five attempts per variant, `STALL_TIMEOUT_S=120`:

| variant | attempts | stalled | phase |
| --- | --- | --- | --- |
| `baseline` | 5 | **1** | `fresh_world_first` |
| `nchannels_16` | 5 | **1** | `fresh_world_first` |
| `nchannels_8` | 5 | 0 | - |
| `net_socket` | 5 | 0 | - |

Pooling every clean observation of `baseline` at 64 ranks — jobs 21790393 (healthy),
21794113 (stalled), 21794114 (healthy), and 1 of 5 here — gives **2 stalls in 8
attempts, about 25%**.

**`NCCL_MAX_NCHANNELS=16` does not fix the stall.** It hung once in five, the same rate
as baseline. This retracts the recommendation drafted after job 21791400, which read the
zero bandwidth cost of a 16-channel cap as making it the obvious thing to try first. It
is free, and it does not work. The bandwidth measurement stands; the inference from it
was wrong.

**And 0 of 5 does not establish that `NCCL_MAX_NCHANNELS=8` works.** At a 20% base rate,
five clean attempts happen by chance 33% of the time (0.8^5). The same applies to
`net_socket`. Both are *consistent* with being fixes and neither is demonstrated. Jobs
21811437 and 21811438 run 15 attempts per variant, where a clean sweep would have a
3.5% chance of being luck — enough to state a conclusion.

This is the reason the harness has a `REPEATS` mode at all. A single healthy run of a
proposed fix, against an intermittent failure, is indistinguishable from no fix.

## Channel-cap dose response, properly powered (job 21811438)

15 attempts per variant, 8 nodes, world size 64:

| variant | attempts | stalled | rate | p(this clean, if the rate were 33%) |
| --- | --- | --- | --- | --- |
| `nchannels_16` | 15 | **5** | 33% | - |
| `nchannels_4` | 15 | **0** | 0% | 0.67^15 = **0.002** |

Two conclusions, one of them a retraction being made final:

- **`NCCL_MAX_NCHANNELS=16` does not fix the stall.** 5 in 15, indistinguishable from
  baseline. The earlier "capping at 16 is free, so try it first" recommendation is
  withdrawn for good. Free and ineffective.
- **`NCCL_MAX_NCHANNELS=4` does fix it**, and this time the statistics support the claim:
  zero stalls in 15 attempts against a 33% background rate has a 0.2% chance of being
  luck. But it costs **56%** of training-band bandwidth (job 21791400), which makes it
  a fix nobody bandwidth-bound would want.

Note that `nchannels_16` at 33% also gives a better estimate of the background rate than
the pooled `baseline` figure did, and it is consistent with it (25% over 8 attempts).

So the dose response is real but sharp: 16 does nothing, 4 works and is ruinous, and 8 —
the reporter's own setting, at 20% bandwidth cost — is the interesting middle. Job
21811437 is measuring it at 15 attempts alongside baseline.

The remaining question worth more than any of these is whether something with **no**
bandwidth cost works. `NCCL_SOCKET_IFNAME` is not a cap: it was healthy in every
observation so far and independently cuts first-collective setup time by ~38%. Job
21812432 puts it and `NCCL_RUNTIME_CONNECT=0` through the same 15-attempt test. If the
interface pin holds at 0/15, it is strictly better than any channel cap and it is the
answer to question 2.

## Q1 and Q2, answered: the interface pin is the fix; capping channels only lowers the odds (jobs 21811437, 21812432)

15 attempts per variant, 8 nodes, world size 64. Bandwidth costs from job 21791400.

| variant | stalled | rate | bandwidth cost | phases seen stalling |
| --- | --- | --- | --- | --- |
| `baseline` | **12/15** | 80% | - | `world_first`, `world_second`, `tp_like_first`, `pp_like_first`, `fresh_world_first` |
| `nchannels_8` | **7/15** | 47% | -20% | `fresh_world_first` |
| `nchannels_16` | 5/15 | 33% | none | `fresh_world_first` |
| `runtime_connect_off` | 4/15 | 27% | none | `fresh_world_first` |
| `nchannels_4` | 0/15 | 0% | **-56%** | - |
| **`socket_ifname`** | **0/15** | **0%** | **none** | - |

### The reporter's workaround is not a fix

`NCCL_MAX_NCHANNELS=8` leaves 7 stalls in 15. It lowers the failure rate from 80% to
47% and costs 20% of training-band bandwidth to do it. That is consistent with the
report — they say they "keep hitting" the stall, not that it always happens — and it
explains why capping felt like a cure: at 47% a few good runs in a row are unremarkable.
It also means the earlier `nchannels_8` result of 0/5 was exactly the luck it was
flagged as being.

### `baseline` stalls in five different phases

`world_first`, `world_second`, `tp_like_first`, `pp_like_first` and `fresh_world_first`
all appear across the 12 stalled attempts. This is the reported symptom reproduced
literally: the stall is not tied to one phase, and clearing one phase does not prevent a
stall in a later one. Note this includes `world_second`, a warm collective — so the
failure can strike a communicator that has already carried traffic successfully, not
only a fresh one. That is broader than the original H-B and it is what makes the symptom
look so erratic from the outside.

### `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` never stalled

Zero stalls in 15 attempts, against a 47% rate for `nchannels_8` measured in the same
week and an 80% rate for `baseline`. Against 47%, a clean sweep of 15 has probability
0.53^15 = 6e-5. Pooled across the whole investigation — jobs 21790359 (2 nodes),
21790392 (4), 21790393 (8), 21794113 (8, run immediately after a 64-rank baseline
stall), and 21812432 (15 attempts) — it is roughly **19 clean observations and zero
stalls**, at three scales.

It also costs nothing: it is not a cap, so the bandwidth curve does not apply, and it
independently *reduces* first-collective setup time by ~38% (15.7 s to 9.8 s).

**One confound, being closed.** In job 21812432 `socket_ifname` ran first and
`runtime_connect_off` second, so the winner was also the least exposed to any residual
state. Its clean record in job 21794113 — where it ran straight after a 64/64 baseline
stall — argues against an ordering artefact, but jobs 21817829 (8 nodes) and 21817831
(4 nodes) settle it properly by running 15 baseline attempts first and then asking
`socket_ifname` to stay clean on those same nodes.

### The stall rate itself varies between jobs

`baseline` at 64 ranks has measured 0/1, 1/1, 0/1, 1/5 and 12/15 across jobs — roughly
25% earlier and 80% here. The ordering of variants has been consistent throughout, so
the likely cause is the node set or concurrent fabric load rather than the harness. The
practical consequence is that **absolute rates from a single job are not portable**;
only within-job comparisons and pooled counts should be quoted. It is also a reason the
reporter's experience may differ run to run and week to week.

## CORRECTION: repeats within a job are not independent, so the p-values above are void

Stall sequences in repeat order (`X` = stalled, `.` = ok, r1 to r15):

| job | variant | sequence | stalls |
| --- | --- | --- | --- |
| 21811437 | `baseline` | `...XXXXXXXXXXXX` | 12/15 |
| 21817829 | `baseline` | `.XX..X.XXXXX..X` | 9/15 |
| 21811437 | `nchannels_8` | `.X.X.X.X.X.X.X.` | 7/15 |
| 21817829 | `socket_ifname` | `....X........X.` | 2/15 |
| 21812432 | `runtime_connect_off` | `....X.XX......X` | 4/15 |
| 21811438 | `nchannels_16` | `..X...XX..X.X..` | 5/15 |

`baseline` in job 21811437 runs clean three times and then stalls **twelve consecutive
times** — an absorbing state, not a coin flip. `nchannels_8` alternates almost perfectly.
Neither is remotely consistent with independent Bernoulli trials.

**Every significance figure quoted earlier in this document is therefore withdrawn**:
`0.8^5 = 0.33` for the 5-attempt runs, `0.67^15 = 0.002` for `nchannels_4`, and
`0.53^15 = 6e-5` for `socket_ifname`. All three assumed independence between attempts.
They do not license any conclusion.

**What this means, in order of importance:**

1. **Attempts within one job share state.** Something persists between attempts that the
   reaping fix does not clear — either leaked fabric/endpoint resources, or genuine
   memory in the node set. The absorbing pattern says a node set can enter a mode where
   it stalls essentially always.
2. **`socket_ifname`'s advantage is confounded with sequence position after all.** It was
   0/15 running first (job 21812432) and 2/15 running second (job 21817829). Both may be
   position effects rather than treatment effects.
3. **`nchannels_4`'s 0/15 is likewise position-advantaged** — checking the `VARIANTS`
   array order, `nchannels_4` precedes `nchannels_16`, so it ran first in job 21811438.
4. **The effect directions still look large and real.** Pooled at 8 nodes: `baseline`
   21/30, `nchannels_8` 7/15, `socket_ifname` 2/30. Those gaps are big enough that
   position alone is unlikely to explain them. But "unlikely" is not a measurement, and
   this document should not contain another number that turns out to rest on a bad
   assumption.

**The correct design, which has not yet been run:** one variant per job, in first
position, one attempt per job. Then each sample gets a fresh allocation and fresh node
state, and samples are genuinely independent. It costs one job per data point, which is
why it was not the first thing tried — but it is the only design that supports a rate
with a confidence interval attached.

Until that runs, the defensible statement is qualitative: **at 64 ranks and default
settings the stall is frequent and can become persistent; pinning `NCCL_SOCKET_IFNAME`
reduces it markedly; capping channels at 8 reduces it less; and no tested setting has
been shown to eliminate it.**

### Scale threshold: nothing stalls at 32 ranks (job 21817831)

4 nodes, world size 32, 15 attempts each: `baseline` 0/15 and `socket_ifname` 0/15. With
the caveat above about independence, 30 consecutive clean attempts at 32 ranks against
frequent stalls at 64 places the onset between the two. That matches the reporter's
8-node freeze, and it means reproduction work must be done at 8 nodes.

### The CXI telemetry collection captured nothing

`cxi_counters.sh` produced 480 snapshot files across 240 before/after pairs and **not one
counter changed**, because `cxi_stat` on LUMI prints device inventory (part numbers, link
state, MAC/NID) rather than counters, and the sysfs paths the script globs for
(`*cq*`, `*eq*`, `*pt_te*`, `*retry*`) yield nothing under `/sys/class/cxi/cxi*` at
depth 3. So hypothesis 5 has no telemetry evidence either way, and the accumulation
hypothesis above cannot be tested with what is currently collected. The counter source
needs to be found before that section of the harness is worth running again.

## FINAL on the fix question: nothing tested prevents the stall (independent samples)

The first attempt executed in a job runs on a freshly allocated node set with nothing
before it to have left state behind, so **one such sample per job is independent**.
Mining every job in this investigation gives, at 8 nodes / world 64:

| variant | independent first attempts | stalled |
| --- | --- | --- |
| `baseline` | 8 | **2 (25%)** |
| `socket_ifname` | 2 | **1** |
| `nchannels_4` | 1 | 0 |

At 4 nodes / world 32, `baseline` first attempts: 0 stalls in 2 jobs (plus 0/15 and
0/15 in repeats).

**`NCCL_SOCKET_IFNAME` does not prevent the stall.** Job 21818931 stalled on its first
attempt on a fresh allocation. Its earlier record — 0/15 running first in job 21812432,
2/15 running second in 21817829, ~19 apparently clean observations — was position and
accumulated state, not a treatment effect. **The Q2 recommendation drafted from that is
withdrawn.**

On independent samples, no setting tested in this investigation has been shown to
prevent the stall, and the sample sizes (8, 2, 1) are too small to rank any of them.
What the interface pin *does* have is a robust, separately measured benefit: it cuts
first-collective setup time from ~15.7 s to ~9.8 s (~38%) consistently across every job
and all three scales. That is a real improvement and a reason to set it. It is not a
cure for the stall, and it should not be presented as one.

### What is actually established

| # | Finding | Strength |
| --- | --- | --- |
| 1 | At 64 ranks and stock defaults, RCCL startup hangs indefinitely on ~25% of fresh allocations (2/8 independent samples), in varied phases including warm collectives | **solid** — reproduced across 8 jobs and several days |
| 2 | Nothing stalls at 32 ranks | **solid** — 2 independent + 30 repeat attempts, zero stalls |
| 3 | `NCCL_NET_GDR_LEVEL=PHB` hangs all ranks deterministically at 4 **and** 8 nodes | **solid, and the strongest causal claim here** — at 4 nodes the background rate is zero, yet this stalls 32/32, first-stall position in its job |
| 4 | Channel-cap bandwidth cost: 20% at 8, 56% at 4, none at 16, all at large messages | **solid** — an independent measurement, unaffected by the stall statistics |
| 5 | `NCCL_SOCKET_IFNAME` cuts first-collective setup ~38% | **solid** — consistent across 3 scales and many jobs |
| 6 | Attempts within one job share state (absorbing and alternating stall sequences) | **solid, and it invalidated our own statistics** |
| 7 | RCCL finds no network path for GCDs 1, 3, 7 in every configuration | **solid**, mechanism unquantified |
| 8 | Any *fix* for the stall | **not established** |

### Why finding 3 is worth reporting regardless of the reporter's case

At 4 nodes the background stall rate is zero across 32 attempts, so `gdr_level` stalling
32/32 there is unambiguous — no accumulated state, no position effect, no race to
confound it. `NCCL_NET_GDR_LEVEL=PHB` is set by the LUMI AI Guide's
`5-experiment-tracking/run_*.sh`. Those lessons are single-node, where it is harmless,
but anyone copying that block into a multi-node job gets a deterministic hang. That is
worth fixing in the guide whether or not it is what the reporter hit.

## Rung 3: vLLM at 8 nodes / world 64 (job 21819544)

`openai/gpt-oss-120b`, TP 8 x PP 8 = world 64, the same geometry as the pure-RCCL hang.
All three variants reached health; **no stall in this allocation**, which at a ~25%
per-allocation rate is the likely outcome and is therefore weak evidence either way.

| variant | time to `/v1/models` | `pynccl_allreduce` | `graph_capture` | `kv_cache` |
| --- | --- | --- | --- | --- |
| `baseline` | **354 s** | 64 s | 186 s | 216 s |
| `socket_ifname` | **124 s** | 56 s | 71 s | 73 s |
| `nchannels_8` | **130 s** | 56 s | 76 s | 78 s |

**A 2.9x difference in real startup time**, and the phase timeline says it is *not* in
communicator setup: `pynccl_allreduce` differs by only 8 s (64 vs 56). The gap opens
later, in graph capture and memory profiling — 186/216 s for baseline against 71/73 s
with the interface pinned. Those phases run real collectives, so slower collectives make
them slower. This is a separate benefit from the setup-time win and a much larger one.

Recorded plainly: this is one allocation per variant, and baseline stall behaviour is
known to vary by allocation, so the 2.9x should be reproduced before being quoted as a
number. The direction is consistent with every other measurement of the interface pin.

### vLLM builds three communicators, each logging the reporter's freeze line

```
(Worker pid=76703) INFO ... [cuda_communicator.py:232] Using ['PYNCCL'] all-reduce ... for group 'tp:0' ...
(Worker pid=76703) INFO ... [cuda_communicator.py:232] Using ['PYNCCL'] all-reduce ... for group 'pp:0' ...
(Worker pid=76703) INFO ... [cuda_communicator.py:232] Using ['PYNCCL'] all-reduce ... for group 'ep:0' ...
```

Three groups — tensor, pipeline and expert parallel — at 64 s, 65 s and 66 s. This
matters for reading the report: **"frozen on `cuda_communicator.py:266 Using ['PYNCCL']
all-reduce`" does not say which communicator was being built**, because the line is
emitted once per group. It also lines up with the pure-RCCL finding that fresh
communicator creation is where hangs occur: vLLM creates at least three in a row, each
one an opportunity to hang. (The line is `:232` in this container against `:266` in the
reporter's, so their vLLM differs in version.)

## Symptom B is almost certainly Lustre, not RCCL (job 21819545)

Raw sequential read of the `Kimi-K2-Instruct-0905` checkpoint on one node, no loader and
no GPU involved:

| | |
| --- | --- |
| Checkpoint size | 1,029,207,342,689 bytes (**1.03 TB**, not 959 GB) |
| Filesystem | `/pfs/lustrep4`, Lustre |
| Read | 48.15 GiB in 35.30 s |
| Throughput | **1.46 GB/s** |
| **Implied floor for the whole checkpoint** | **703 s (11.7 min)** |

The reporter's second symptom is the weight streamer sitting at `0% Completed` for
**850+ s**. The floor imposed by Lustre read bandwidth alone is **703 s**. Those are the
same number to within the precision of this measurement.

Two supporting details:

- The string they quote is literally a tqdm progress bar:
  `Loading safetensors checkpoint shards:   0% Completed | 0/15`. It shows 0% until the
  *first shard* finishes, so a long period at 0% is what slow-but-working I/O looks
  like, not evidence of a hang.
- The launchers read weights from `/scratch` (Lustre) and never from `/flash` (NVMe) —
  F9 — and `launch_vllm_multinode_rank.sh:27-28` defaults
  `RUNAI_STREAMER_CONCURRENCY=1`, which throttles the reads further when the streamer
  is actually enabled.

**So symptom B is a different problem from symptom A, with a different fix**: stage the
checkpoint to `/flash`, raise `RUNAI_STREAMER_CONCURRENCY`, and expect ~12 minutes of
unavoidable read time for a 1 TB checkpoint on this filesystem. Hypotheses 7 and 8 are
resolved in favour of 8. This also means capping channels never had anything to do with
symptom B, and the apparent link in the report is coincidence.

## Hypotheses

Ordered by prior probability. Every row must resolve.

| # | Hypothesis | Decided by | Status |
| --- | --- | --- | --- |
| 1 | `NCCL_SOCKET_IFNAME` unset → RCCL uses the wrong interfaces (**H-A, revised**) | jobs 21790359, 21811437, 21812432 | **CONFIRMED in its revised form**: not a slow bootstrap (`init` is 0.76 s either way) but wrong data-path interface selection. Pinning it fixes the stall (0/15) and cuts setup 38% |
| 2 | New communicators are where multi-node RCCL fails (**H-B, revised**) | job 21794113 — `baseline` hangs 64/64 in `fresh_world_first` after a healthy first collective | **CONFIRMED as the failure site**, not as a cost: it is an intermittent hang, not a slow burst. Stall rate pending job 21811023 |
| 3 | Channel count governs whether the hang happens | job 21811438 | **confirmed, sharply**: cap 16 -> 5/15 stalls, cap 4 -> 0/15. Rank-count scaling separately refuted (`world_first` flat at 14.71/15.71/15.82 s for 16/32/64 ranks) |
| 4 | `NCCL_NET_GDR_LEVEL` unset degrades the chosen path | rung 2 `gdr_level` variant | _pending_ |
| 5 | CXI endpoint/resource exhaustion during setup, not CQ depth | job 21794113 — `cxi_cq_and_sw_match` healthy; job 21794114 — `NCCL_NET=Socket` rescues the hang | **the hang is in the OFI/CXI path**, but CQ size and match mode are not the lever, matching the reporter's null result |
| 6 | Missing CPU/NUMA/NIC affinity | job 21794113 — `cpu_bind` stalled in the same phase as `baseline` | **no evidence it helps**; likely just another sample of the same race, needs repeats |
| 7 | Symptom B is the RunAI streamer throttle, not RCCL | rung 7, single node | _pending_ |
| 8 | Symptom B is Lustre read bandwidth on the (actually 1.03 TB) checkpoint | job 21819545 — 1.46 GB/s, floor 703 s vs reported 850 s | **CONFIRMED as the explanation for symptom B** |
| 9 | Per-job MIOpen cache re-pays kernel compilation — a startup cost, not the stall | rungs 4-5 `graph_capture` gap | _pending_ |
| 10 | Container/plugin version — expected **negative**, since the reporter says every image reproduces | rung 2 `container_older` variant | _pending_ |
| 11 | RCCL finds no network path for GCDs 1, 3, 7, forcing their traffic through peer GCDs | job 21790359 — 64 warnings per variant, identical across all three | **confirmed present at defaults**; effect on the stall not yet quantified |

## Verdicts

| job id | rung | nodes / world | what it was for | result |
| --- | --- | --- | --- | --- |
| 21790359 | 1 | 2 / 16 | baseline, `socket_ifname`, `nchannels_8` | all FAST, no stall; cost concentrated in `world_first` |
| 21790392 | 2 | 4 / 32 | full variant table | Valid: baseline and `socket_ifname` healthy, `gdr_level` STALL 32/32. Rows after the first stall retracted |
| 21794113 | 2 | 8 / 64 | clean re-run, control passed | **reproduced the reporter's stall at defaults**: `baseline` 64/64 in `fresh_world_first` |
| 21794114 | 2 | 8 / 64 | GDR rescue combos | `NCCL_MAX_NCHANNELS=8` and `NCCL_NET=Socket` rescue it; eager connect and interface pin do not |
| 21819544 | 3 | 8 / 64 | vLLM `gpt-oss-120b` startup | all READY; baseline 354 s vs `socket_ifname` 124 s; 3 communicators (tp/pp/ep) |
| 21819545 | 7 | 1 | Lustre read floor, Kimi 1.03 TB | 1.46 GB/s -> 703 s floor, matching the reported 850 s |
| 21818930 | 2 | 8 / 64 | independent single attempt, baseline | ok |
| 21818931 | 2 | 8 / 64 | independent single attempt, `socket_ifname` | **STALL** — the pin is not a fix |
| 21811023 | 2 | 8 / 64 | 5x repeats of baseline and caps | baseline 1/5, `nchannels_16` 1/5, `nchannels_8` 0/5, `net_socket` 0/5 — underpowered |
| 21811437 | 2 | 8 / 64 | 15x baseline and `nchannels_8` | baseline 12/15 across five phases; `nchannels_8` 7/15 — a probability reduction, not a fix |
| 21811438 | 2 | 8 / 64 | 15x `nchannels_16` and `nchannels_4` | 16 stalls 5/15 (no fix, final); 4 stalls 0/15 (real fix, but -56% bandwidth) |
| 21812432 | 2 | 8 / 64 | 15x `socket_ifname` and `runtime_connect_off` | **`socket_ifname` 0/15**; `runtime_connect_off` 4/15 |
| 21817829 | 2 | 8 / 64 | baseline then `socket_ifname`, 15x each | baseline 9/15, `socket_ifname` **2/15** — the pin is not a cure; also revealed non-independence |
| 21817831 | 2 | 4 / 32 | same at 4 nodes | 0/15 and 0/15 — no stalls at 32 ranks |
| 21791400 | 5 | 8 / 64 | bandwidth vs channel cap | cap 16 free, cap 8 costs 20%, cap 4 costs 56% in the training band |
| 21790393 | 2 | 8 / 64 | full variant table | **mostly invalid**: positive control `net_socket` stalled. Valid: baseline healthy, `socket_ifname` healthy, `gdr_level` STALL 64/64. Rest retracted |

## Answers to the four questions asked

Each answer must cite the job ids that support it. Left blank deliberately until they
exist.

**1. Known LUMI behaviour, or their misconfiguration?**
**Both, and the honest answer names our share of it.** It is default-configuration
behaviour on LUMI: this repo sets no comms variable anywhere (F1), the bindings module
sets none (F2), and at those defaults `baseline` stalls 12 times in 15 at 64 ranks
(job 21811437). Nothing they did caused it. But it is *also* a configuration gap that is
fixable from the user side, and the missing setting — `NCCL_SOCKET_IFNAME` — is absent
from this repo and from the guide's multi-node lesson alike (F5). So it is our gap as
much as theirs, and F3 shows we were already paying for it: our own README needs startup
timeouts of 2700-14400 s for multi-node launches, which is this stall, undiagnosed.

**2. Is there a better fix than capping channels?**
**Not found. And capping channels is not a fix either.** On independent samples nothing
tested prevents the stall: `socket_ifname` stalled 1 of 2 fresh allocations,
`nchannels_8` leaves 7/15 within a job, `nchannels_16` 5/15. The apparent winners in
the repeat runs were position artefacts. Two things are still worth setting on their own
merits — `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` for a ~38% cut in first-collective
setup time at no bandwidth cost, and *not* setting `NCCL_NET_GDR_LEVEL` — but neither is
a cure. Practical advice for the reporter: keep the generous startup timeouts, keep
`NCCL_MAX_NCHANNELS=8` only if they measure it helping their own workload and can afford
20% on bandwidth-bound jobs, and treat the underlying hang as an open platform issue
rather than a configuration mistake on their side.

**3. What does capping to 8 cost in collective bandwidth?**
**Answered** (job 21791400, 8 nodes / world 64). About **20%** of bus bandwidth in the
128 MiB - 1 GiB band that bandwidth-bound training uses — all_reduce falls from
87.6 to 70.5 GB/s. Capping at **4** costs 56%. Capping at **16** costs nothing
measurable. Small messages (<= 1 MiB) are unaffected by any cap, so an inference server
is largely insensitive while a training job pays in full — the reporter's instinct that
this is the wrong knob for bandwidth-bound training is correct.

Note the tempting inference that does **not** hold: a 16-channel cap is free, but job
21811023 shows it does not stop the stall (1/5, same as baseline). Free and ineffective.
So on current evidence the choice really is between paying ~20% for a cap of 8, or
finding a different fix — which is what makes question 2 worth pursuing rather than
settling for the workaround.

**4. Recommended `NCCL_*` / `FI_CXI_*` baseline?**
_pending_ — `env_baseline.sh`. F6 means there is no existing measured baseline to point
at, so this investigation has to produce one. Every line must carry a job id and a
stated cost before it is uncommented.

## Corrections to the original report

| Claim in the report | What the runs show |
| --- | --- |
| _pending_ | |

## Corrections to our own prior claims

Recorded here as they arise, not deleted. F3 is the first candidate: the root README
presents its multi-node recipes as "Successful Launch Commands" without noting that
several needed startup timeouts of 45 minutes to 4 hours, which in hindsight is the
same symptom being reported.

## Open questions for the LUMI AI Guide

- F5: should `NCCL_SOCKET_IFNAME` and `NCCL_NET_GDR_LEVEL` move into, or be duplicated
  in, `3-multi-gpu-and-node/`? They currently appear only in lesson 5, which is
  single-node.
- F10: should the persistent per-user MIOpen cache path from `setup.sh` be adopted by
  this repo's launchers?
