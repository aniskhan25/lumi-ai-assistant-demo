# Draft: issue for HewlettPackard/shs-ccl-docs

**NOT SENT.** Public, and it contradicts a vendor guide, so every claim has to resolve
to a job id in `FINDINGS.md` before this goes anywhere. Sections marked `<pending>` are
this study's results and must not be sent while they are placeholders.

Anything already measured is from the sibling investigation in
`../rccl_startup_gfx90a/`, which is complete and whose job ids are real.

---

**Title:** RCCL tuning guide: two recommendations are harmful or inert on LUMI (MI250X + Slingshot 11)

Hello — we run a large MI250X / Slingshot 11 system (LUMI, 4× MI250X = 8 GCDs and 4
`hsn` NICs per node) and have been measuring `rccl/rccl_tuning_guide.md` against it.
Most of it holds. Two items do not, and one omission looks significant on this
hardware. All numbers below are from single-variant Slurm allocations — we do not
compare variants that ran sequentially inside one job, for reasons in the last section.

### 1. `NCCL_NET_GDR_LEVEL=PHB` hangs deterministically here

The guide gives this as "Required to enable RDMA between GPUs". On LUMI it hangs the
**first cross-node collective on every rank**, deterministically:

| nodes | ranks hung | jobs |
| --- | --- | --- |
| 4 | 32/32 | 21790392, 21790393 |
| 8 | 64/64 | 21794114 |

At 4 nodes our background stall rate is zero, so this is unambiguous rather than an
intermittent fault being blamed on the variable. It is tracked downstream as
`lumi-ai-factory/laifs-container-recipes#30`.

Worth noting separately: **ROCm ≥ 6.2 already defaults to this behaviour**, so setting
it explicitly buys nothing even on systems where it does not hang. We would suggest
either dropping it or qualifying it with a ROCm version range.

### 2. Ten of the eleven RCCL-list variables are inert here; one does all the work

Measured at 4 nodes with 8 communicators (job 21838977):

| variant | stalls |
| --- | --- |
| the full HPE set | 0/5 |
| the same set with `FI_MR_CACHE_MONITOR` **removed** | 4/5 |
| baseline, nothing set | 4/5 |

So on LUMI `FI_MR_CACHE_MONITOR` accounts for the entire benefit and the remaining ten
variables change nothing we can measure. `FI_CXI_DISABLE_HOST_REGISTER=1` on its own
leaves 3/5 stalls (job 21838111).

We are not suggesting the other ten are wrong for Slingshot generally — they may well
matter on other machines or other workloads. But the guide presents them as a set, and
on this hardware a reader who adopts the set gets the benefit of one variable and nine
unexamined ones, one of which is item 1 above.

### 3. The rationale for `FI_MR_CACHE_MONITOR` understates it

The guide says it "Sets the memory cache monitor to detect changes between virtual and
physical memory pages". The mechanism we measured is sharper and more useful to a
reader (job 21844179): libfabric defaults to `memhooks` here, which detects remapping
by intercepting userspace allocator calls and **does not reliably see ROCm memory
operations**, so a stale registration is never invalidated and the RDMA silently never
completes. Setting each monitor explicitly against a reliable reproducer:

| monitor | stalls |
| --- | --- |
| `memhooks` (the default here) | 4/5 |
| `userfaultfd` | 0/5 |
| `kdreg2` | 0/5 |
| `disabled` | 0/5 |

`kdreg2` works as well as `userfaultfd`, which the guide mentions only in passing.

### 4. Omission: `HSA_NO_SCRATCH_RECLAIM` on MI200

AMD's own RCCL documentation reports a 5–10× small-message latency penalty on gfx90a
without `HSA_NO_SCRATCH_RECLAIM=1`. The guide does not mention it. `<pending: our own
B0/B1 latency A/B, with job ids — do not send this section until measured>`

### 5. The guide has no tuning content

This is meant as a scoping observation rather than a complaint. The document is a
correctness and configuration reference: it contains no benchmarking methodology and no
performance knob at all — no channel counts, no `NCCL_ALGO`/`NCCL_PROTO`, no
`NCCL_BUFFSIZE`, no MSCCL. Given the filename, readers arrive expecting tuning and
leave with configuration. Either renaming it or adding a short "this guide does not
cover performance tuning" line near the top would save people the trip.

### Methodology, since it changes how much the above is worth

Every comparison above is **one variant per Slurm allocation**. We learned the hard way
that variants run sequentially inside one allocation are not independent: the first run
pays one-time costs later runs do not, and a stalled attempt leaves ranks blocked inside
RCCL holding their GCDs, which poisons whatever runs next. That single flaw produced
five findings we had to retract, including an apparent 2.9× win and an apparent complete
fix, both of which evaporated under single-variant allocations. Every sweep carries a
pre-registered control that must not fail.

Happy to provide the harness, the raw JSON, or to re-run anything at a different scale.
