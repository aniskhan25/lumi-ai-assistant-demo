# Reply to the multi-node vLLM startup stall report

Draft reply to the LUMI support thread. Published as a shareable page at
https://claude.ai/code/artifact/0438c090-3b54-4611-8379-9c7bf82b3ff2

Every figure below carries a Slurm job id in `FINDINGS.md`, including the withdrawn ones.
~20 jobs, ~300 GPU-hours. RCCL 2.26.6, ROCm 7.0.2, torch 2.10.0+rocm7.0.

## Answers

**1. Known LUMI behaviour, or your misconfiguration?** Platform behaviour at defaults.
This repo sets no `NCCL_*`/`RCCL_*`/`FI_CXI_*` variable anywhere, the
`lumi-aif-singularity-bindings` module sets only `SINGULARITY_BIND` and
`SLURM_MPI_TYPE`, and the container sets none at all — and the hang still reproduces.
Nothing you configured causes it. We were already paying for it too: our own multi-node
recipes carry `STARTUP_TIMEOUT_S` of 2700-14400 s, which is this stall, undiagnosed.

**2. Better fix than capping channels?** None found, and capping channels is not a fix
either — `NCCL_MAX_NCHANNELS=8` still hung 7 of 15 attempts. The one firm recommendation
is negative: **do not set `NCCL_NET_GDR_LEVEL=PHB`**.

**3. Cost of capping to 8?** ~20% of all-reduce bus bandwidth in the 128 MiB - 1 GiB band
(87.6 -> 70.5 GB/s at 8 nodes). Cap 4 costs 56%. Cap 16 costs nothing but does not stop
the stall. Small messages (<= 1 MiB) unaffected at every cap, so inference is largely
insensitive while training pays in full. Your instinct about the wrong knob was right.

**4. Recommended NCCL/FI_CXI baseline?** Not one we can justify. Every positive candidate
we tested turned out to be a measurement artefact. No tuned RCCL baseline for LUMI exists
yet, and we would rather say that than ship settings whose benefits we could not
reproduce.

## What reproduced

No vLLM and no weights needed — a bare `torch.distributed` probe timing each startup
phase is enough. At 8 nodes / world 64, stock settings, on **independent samples** (one
variant per job, first position):

| scale | world | fresh allocations | hung | phases seen hanging |
| --- | --- | --- | --- | --- |
| 8 nodes | 64 | 8 | **2** | `world_first`, `world_second`, `tp_like_first`, `pp_like_first`, `fresh_world_first` |
| 4 nodes | 32 | 32 | 0 | - |

The varied phases — including `world_second`, a repeat all-reduce on a communicator that
already carried traffic — are your symptom exactly: clearing one stage does not prevent a
stall at a later one. Onset is between 32 and 64 ranks.

**Your freeze line does not identify which communicator.** vLLM builds three in
succession, each logging it (`tp:0`, `pp:0`, `ep:0`, at 64/65/66 s in our run). Ours logs
at `cuda_communicator.py:232` against your `:266`, so our vLLM versions differ.

## Your second symptom is a different bug

The streamer at `0% Completed` for 850+ s is Lustre, not RCCL:

| | |
| --- | --- |
| `Kimi-K2-Instruct-0905` checkpoint | 1.03 TB |
| Read throughput from `/pfs/lustrep4` | 1.46 GB/s |
| **Implied floor for the full read** | **703 s** |
| Time you reported at 0% | 850 s |

The string is a tqdm bar (`Loading safetensors checkpoint shards: 0% Completed | 0/15`)
that shows 0% until the *first shard* finishes, so a long spell at 0% is slow-but-working
I/O. **This one has a fix**: weights are read from `/scratch` (Lustre) and never
`/flash` (NVMe), and the multi-node launcher defaults `RUNAI_STREAMER_CONCURRENCY=1`.
Stage to `/flash`, raise concurrency, and expect ~12 min of unavoidable read for a 1 TB
checkpoint. Capping channels never had anything to do with this symptom.

## Findings, with confidence

| finding | status |
| --- | --- |
| Hang at 64 ranks, stock settings, varied phases (2/8 fresh allocations) | confirmed |
| Nothing hangs at 32 ranks (32 attempts) | confirmed |
| `NCCL_NET_GDR_LEVEL=PHB` hangs every rank deterministically (32/32 at 4 nodes, 64/64 at 8) | confirmed |
| Channel-cap bandwidth cost: 20% / 56% / none | confirmed |
| Weight-loader symptom is Lustre read time | confirmed |
| Cold-start penalty ~167 s per 8-node vLLM launch | confirmed |
| RCCL finds no network path for GCDs 1, 3, 7 | unquantified |
| **Any setting that prevents the hang** | **not found** |

## Suggested actions

- **Keep the generous startup timeouts.** The hang is real and unfixed.
- **Check whether anything in your environment sets `NCCL_NET_GDR_LEVEL`.** If it does,
  that alone explains your report end to end. Nothing on the platform sets it, so it
  would come from your own scripts. This is the single most useful thing you can tell us.
- **Treat the weight-loader symptom separately**: `/flash`, higher streamer concurrency.
- **Keep `NCCL_MAX_NCHANNELS=8` only if you can measure it helping your workload** and can
  afford 20% on bandwidth-bound jobs. It lowers the hang rate; it does not remove it.

Your `NCCL_NET=Socket` observation matches everything we measured — it was also the only
thing that reliably avoided a hang here, at a throughput cost you have already correctly
rejected.

## Still open

- No fix. Needs escalation to the RCCL or libfabric/CXI layer, not more env-var search.
- Why attempts within one allocation share state — probably our most informative
  unexplained observation. Our CXI telemetry captured nothing usable: `cxi_stat` reports
  device inventory, not counters, so the counter source still has to be found.
- Whether the missing network path for GCDs 1, 3, 7 contributes.

## Method note, worth passing on

Five of our own conclusions were withdrawn during this investigation, four of which
looked convincing enough to send. All five had one cause: **on this system any comparison
between variants run sequentially in one allocation is worthless**, because the first run
pays one-time costs (MIOpen kernel compilation, Lustre first-touch, and whatever RCCL
caches between communicators) that later runs do not. Only one-variant-per-allocation
comparisons mean anything.

The withdrawn claims, in order: an 8-node sweep of 11 "findings" that a pre-registered
control exposed as zombie ranks poisoning later variants; "cap 16 is free so try it
first" (free, but 5/15 hangs); every p-value we quoted (hang sequences read
`...XXXXXXXXXXXX` and `.X.X.X.X.X.X.X.` — attempts are not independent);
`NCCL_SOCKET_IFNAME` prevents the hang (0/15 in one job, then hung on its first fresh
allocation); and the interface pin makes startup 2.9x faster (baseline 290/123/121 s vs
pin 287/123/139 s — identical cold/warm curves).

See `FINDINGS.md` for the full ledger with job ids.
