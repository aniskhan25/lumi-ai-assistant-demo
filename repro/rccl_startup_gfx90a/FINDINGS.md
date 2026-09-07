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

## Hypotheses

Ordered by prior probability. Every row must resolve.

| # | Hypothesis | Decided by | Status |
| --- | --- | --- | --- |
| 1 | `NCCL_SOCKET_IFNAME` unset → RCCL bootstrap on a non-HSN interface (**H-A**) | job 21790359 — `init` is 0.76 s with or without the pin | **refuted as stated at 2 nodes**; revised form (interface choice affects data-path connection setup, -33% on `world_first`) still open |
| 2 | RCCL lazy connection setup makes each new communicator pay a fresh burst (**H-B**) | job 21790359 — `fresh_world_first` 1.43 s vs `world_first` 14.71 s | **strong form not supported at 2 nodes**; setup is mostly one-time, per-comm increment ~1.3 s. Open whether it grows with ranks |
| 3 | Burst cost scales with `channels x ranks`, so 8 is a dose effect and not a threshold | jobs 21790392 (4n), 21790393 (8n) | _pending_; at 2 nodes `nchannels_8` cut `world_first` 35% with no stall to fix |
| 4 | `NCCL_NET_GDR_LEVEL` unset degrades the chosen path | rung 2 `gdr_level` variant | _pending_ |
| 5 | CXI endpoint/resource exhaustion during setup, not CQ depth | rung 2 + `cxi_*` counter deltas | _pending_ |
| 6 | Missing CPU/NUMA/NIC affinity | rung 2 `cpu_bind` variant | _pending_ |
| 7 | Symptom B is the RunAI streamer throttle, not RCCL | rung 7, single node | _pending_ |
| 8 | Symptom B is Lustre read bandwidth on a 959 GB checkpoint | rung 7 `lustre_read.sh` | _pending_ |
| 9 | Per-job MIOpen cache re-pays kernel compilation — a startup cost, not the stall | rungs 4-5 `graph_capture` gap | _pending_ |
| 10 | Container/plugin version — expected **negative**, since the reporter says every image reproduces | rung 2 `container_older` variant | _pending_ |
| 11 | RCCL finds no network path for GCDs 1, 3, 7, forcing their traffic through peer GCDs | job 21790359 — 64 warnings per variant, identical across all three | **confirmed present at defaults**; effect on the stall not yet quantified |

## Verdicts

| job id | rung | nodes / world | what it was for | result |
| --- | --- | --- | --- | --- |
| 21790359 | 1 | 2 / 16 | baseline, `socket_ifname`, `nchannels_8` | all FAST, no stall; cost concentrated in `world_first` |
| 21790392 | 2 | 4 / 32 | full variant table | _running_ |
| 21790393 | 2 | 8 / 64 | full variant table | _running_ |

## Answers to the four questions asked

Each answer must cite the job ids that support it. Left blank deliberately until they
exist.

**1. Known LUMI behaviour, or their misconfiguration?**
_pending._ The rule for deciding: if this repo — which carries no comms tuning at all
(F1, F2) — stalls at defaults, it is platform default behaviour and not the reporter's
mistake. If `socket_ifname` or `cpu_bind` alone removes the stall, it is a configuration
gap, and F5 says it is one **we and the guide's own multi-node lesson share**, so it
should be reported as ours too rather than pinned on them. F3 is part of the honest
answer either way.

**2. Is there a better fix than capping channels?**
_pending._

**3. What does capping to 8 cost in collective bandwidth?**
_pending_ — from `results/bandwidth.md`, headline the 128 MiB - 1 GiB training band, and
report small-message latency separately.

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
