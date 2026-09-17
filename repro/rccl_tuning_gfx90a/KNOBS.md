# The knob catalogue

Every candidate, where it comes from, what it is expected to do, and — the column
that matters — **how it could silently do nothing**. A knob that is never read
produces a clean null indistinguishable from a real one, so nothing enters the screen
until Stage 1 has given it a row in `knob_verified.md`.

18 candidates reduce to 12 screened factors. The triage is recorded here rather than
in a commit message, because "we did not test that" and "we tested that and it did
nothing" are different claims and the difference has to survive.

## Screened — the 12 PB-20 factors

| # | factor | low → high | source | hypothesis | expected band | how it could silently do nothing |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `HSA_NO_SCRATCH_RECLAIM` | unset → `1` | AMD RCCL usage tips | AMD reports 5–10× small-message latency on MI200 without it | B0, B1 | **No log line and no fd.** Read by the HSA runtime, not RCCL. Container is ROCm 7.0, AMD documents the knob for 7.13+, so it may be a no-op here. `no-proxy`: the latency A/B is the only evidence |
| 2 | `FI_MR_CACHE_MONITOR` | `userfaultfd` → `kdreg2` | `env_baseline.sh:38` | both fix the hang; kdreg2 is a kernel module and may cost less than userfaultfd's page-fault path on registration-heavy traffic | B1, B2 | Behavioural check works: userfaultfd shows as an fd, kdreg2 as `/dev/kdreg2`. Proven in the bug study |
| 3 | MSCCL++ group | off → `RCCL_MSCCLPP_ENABLE=1` + threshold | LUMI tips-and-tricks | off by default on non-MI300X; may help small all-reduce | B0, B1 | **Most likely to be a no-op.** Support may be compiled out of this container entirely. If Stage 1 finds no "MSCCL … N algorithms" line the factor is **dropped**, not carried as a null |
| 4 | `NCCL_MIN_NCHANNELS` | unset → `32` | AMD RCCL usage tips | the bug study only ever *capped* channels; raising the floor is the untested direction | B1, B2 | INIT trace prints the channel count |
| 5 | `NCCL_MAX_NCHANNELS` | unset → `16` | the reporter's workaround | capping costs 20% of bulk at 8 and 56% at 4 (jobs 21791400, 21838863); 16 may be free | B2, B3 | INIT trace prints the channel count |
| 6 | `NCCL_NCHANNELS_PER_NET_PEER` | unset → `2` | `env_baseline.sh:90` | narrower than a global cap: leaves intra-node collectives at full width | B2 | INIT trace |
| 7 | `NCCL_BUFFSIZE` | unset (4 MiB) → 8 MiB | NCCL | affects pipelining depth; interacts with channel count | B2 | RCCL echoes the buffer size |
| 8 | `NCCL_CROSS_NIC` | unset → `1` | HPE guide | HPE says "improves performance on large systems"; never measured on LUMI | B2, B3 | Weak log evidence; may need a behavioural read |
| 9 | `NCCL_NET_GDR_READ` | unset → `1` | NCCL | GDR on the read path | B2 | Weak log evidence |
| 10 | `NCCL_IGNORE_CPU_AFFINITY` | unset → `1` | AMD RCCL usage tips | lets RCCL use GPU affinity only, ignoring the mask Slurm imposed | B0, B1 | **`no-proxy`.** Nothing reports it back |
| 11 | CXI rendezvous group | default → HPE set | HPE guide | `RDZV_THRESHOLD=0` forces *every* message through rendezvous. Measured inert for the hang, never measured for latency. **Expected to cost in B0/B1** | B0, B1 | `FI_CXI_RDZV_PROTO=alt_read` needs driver property `rdzv_get_en=0` and **falls back silently** without it. If neither the log nor the curve discriminates, drop it as untestable from userspace rather than reporting "no effect" |
| 12 | CPU-bind mask | guide masks → none | `run_rccl_probe.sh:57` | affinity on a 4-NUMA node should matter a lot | all | Applied as an `srun` flag, so it cannot fail to take effect. **But**: the vLLM path runs 1 task/node where the mask is moot, so a win here is microbenchmark-only until the serving launcher changes |

## Pulled out of the screen, handled as grids

| knob | why | where |
| --- | --- | --- |
| `NCCL_ALGO` × `NCCL_PROTO` | multi-level and strongly interacting. Forcing either globally overrides RCCL's per-size tuner, which is usually a loss; the only real question is whether the tuner picks wrong at the decode size | Stage 3a, full 3×3 |
| `NCCL_MAX_NCHANNELS` × `NCCL_BUFFSIZE` | the one interaction PB deliberately does not resolve | Stage 3b, full 3×3 |

## Dropped before the screen

| knob | why | evidence |
| --- | --- | --- |
| `FI_CXI_DEFAULT_CQ_SIZE` | two independent nulls. Spending a design column on a third is how a study runs out of budget | the reporter measured nothing (`env_baseline.sh:92-96`); the probe's own `cxi_cq_and_sw_match` row measured nothing |
| `FI_CXI_RX_MATCH_MODE` | as above | as above |
| `NCCL_P2P_LEVEL` | same foot-gun family as `NCCL_NET_GDR_LEVEL=PHB`, which hangs deterministically. Screened only if Stage 1 shows it is safe | jobs 21790392, 21790393, 21794114 |
| `NCCL_SOCKET_IFNAME` | refuted three times against independent single-variant allocations | jobs 21818930, 21818931, 21822747, 21822748 |
| `HSA_FORCE_FINE_GRAIN_PCIE` | part of HPE's set, measured inert as a group; not separable at this budget and no mechanism suggests it matters on LUMI-G's xGMI topology | job 21838977 |

## Never set

| knob | why |
| --- | --- |
| `NCCL_NET_GDR_LEVEL=PHB` | hangs the first cross-node collective on every rank, deterministically, at 4 nodes (32/32) and 8 nodes (64/64). Comes from HPE's `ccl_env.sh`; tracked in laifs-container-recipes#30. ROCm ≥ 6.2 already defaults to this behaviour, so it buys nothing even where it does not hang |

## Disposition

Filled in as stages complete. `UNVERIFIED` is a disposition, not a gap.

| factor | Stage 1 verified | Stage 2 effect | Stage 4 effect | disposition | job ids |
| --- | --- | --- | --- | --- | --- |
| `scratch_reclaim` | | | | pending | |
| `mr_monitor` | | | | pending | |
| `mscclpp` | | | | pending | |
| `min_nchannels` | | | | pending | |
| `max_nchannels` | | | | pending | |
| `nchannels_per_peer` | | | | pending | |
| `buffsize` | | | | pending | |
| `cross_nic` | | | | pending | |
| `net_gdr_read` | | | | pending | |
| `ignore_cpu_affinity` | | | | pending | |
| `cxi_rdzv` | | | | pending | |
| `cpu_bind` | | | | pending | |
