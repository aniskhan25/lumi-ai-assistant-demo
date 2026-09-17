# Draft: report for lumi-ai-factory/laifs-container-recipes

**NOT SENT.** `<pending>` sections are this study's results. Do not send while they are
placeholders — the point of this report is that it is the first LUMI RCCL baseline with
a job id on every line, and a placeholder would undo that.

---

**Title:** A measured RCCL baseline for LUMI gfx90a — what to set, what is inert, what hangs

Following up on #30 and #44. We have been measuring RCCL environment variables on LUMI
rather than inheriting them, because the official LUMI AI Guide sets no `NCCL_*`,
`RCCL_*` or `FI_*` variable in any lesson — its multi-node lesson runs at stock
defaults, which is exactly where the intermittent hang lives — and HPE's guide turns out
to be a Slingshot-generic configuration list rather than a tuning reference.

### What to set

```sh
export FI_MR_CACHE_MONITOR=userfaultfd
```

0 stalls in 18 attempts across 4 and 8 nodes, against a baseline that hung 21 of 23
(jobs 21838111, 21838977, 21838978). It costs nothing measurable: uncapped all-reduce
bus bandwidth is 88.1 GB/s with it against 87.6 GB/s without (jobs 21838863, 21791400)
— unlike `NCCL_MAX_NCHANNELS=8`, which also stops the hang but costs 21% of
training-band bandwidth.

This is a **workaround** for an open RCCL/libfabric bug (#44), not a root-cause fix.
Keep generous startup timeouts until that is closed.

`kdreg2` fixes it equally well (0/5) and may cost less than `userfaultfd`'s page-fault
path. `<pending: the registration-overhead comparison, which nobody has ever made>`

### What not to set

- `NCCL_NET_GDR_LEVEL=PHB` — deterministic hang, 32/32 ranks at 4 nodes and 64/64 at 8
  (jobs 21790392, 21790393, 21794114). This is #30. It comes from HPE's `ccl_env.sh`.
  Note ROCm ≥ 6.2 defaults to this behaviour anyway, so it buys nothing regardless.
- `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3` — recommended to us three times and refuted
  three times. Against independent single-variant allocations: first cross-node
  collective 15.75 s vs 15.73 s baseline (jobs 21818931, 21818930), vLLM startup
  287/123 s vs 290/123 s cold/warm (jobs 21822748, 21822747). Every apparent win was
  position in the job or cache warming.
- HPE's other nine RCCL variables — measured inert as a group at 4 nodes (job 21838977).

### What is worth measuring next, and what we found

`<pending: the whole tuning study — Stage 0 noise floor, the PB-20 screen, the
algo×proto and channels×buffsize grids, Stage 4 confirmation, Stage 5 end-to-end. Each
line to quote its Stage-4 job id and its measured cost.>`

### The methodological note that matters most for this tracker

If you are benchmarking environment variables on LUMI: **run one variant per Slurm
allocation**, and never compare variants that ran sequentially inside one job. The first
run in an allocation pays one-time costs later runs do not — MIOpen kernel compilation,
Lustre first touch of the weights, whatever RCCL caches between communicators. Measured
on 8 nodes: first vLLM startup ~290 s against ~123 s for later runs in the same job;
first RCCL collective ~15.7 s against ~9.2 s. Worse, a stalled attempt leaves surviving
ranks blocked inside RCCL holding their GCDs, which poisons the next variant and
counterfeits an intermittent bug.

That single flaw produced five false findings in our hang investigation. Put a
pre-registered control in every sweep that must not fail, and reap leftover processes on
every node between attempts.

We are happy to contribute the harness if that is useful to the tracker.
