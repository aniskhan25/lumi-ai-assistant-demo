# Draft comment for lumi-ai-factory/laifs-container-recipes#44

Status: verified against the issue's own reproducer (job 21845322). Ready to post.

---

We hit this independently via multi-node vLLM startup and traced it to the libfabric
memory-registration cache monitor. **Your reproducer, unmodified, hangs 3/5 at default
and 0/5 with `FI_MR_CACHE_MONITOR=userfaultfd`** — 4 nodes / 32 ranks, `timeout 180` per
attempt as in your script:

| condition | hung | note |
| --- | --- | --- |
| default | **3/5** | hung attempts stopped after 160 and 64 `collective done` lines, i.e. at group 6 and group 3 — matching your "which group it stalls on varies" |
| `FI_MR_CACHE_MONITOR=userfaultfd` | **0/5** | all 256 lines (32 ranks x 8 groups) every time |

**`FI_MR_CACHE_MONITOR` defaults to `memhooks` here, and that is the culprit.** Setting
each monitor explicitly, same container as yours, 4 nodes / 32 ranks, 8 successive
`new_group()` + `all_reduce` cycles, 5 attempts each:

| `FI_MR_CACHE_MONITOR` | hung | how it detects remapping |
| --- | --- | --- |
| unset (default) | 3/5 | whichever libfabric picks |
| `memhooks` | **4/5** | intercepts userspace malloc/free/mmap |
| `userfaultfd` | **0/5** | kernel userfaultfd notification |
| `kdreg2` | **0/5** | HPE kernel module |
| `disabled` | **0/5** | no MR caching at all |

`memhooks` is the only monitor that works by intercepting userspace allocator calls, and
it is the only one that hangs. `disabled` being clean says the MR cache itself is the
mechanism. So ROCm/HIP memory operations appear invisible to allocator interception: a
registration cached for a since-remapped buffer is never invalidated, the RDMA targets a
stale registration, and the transfer silently never completes — which matches your
observation that the job blocks indefinitely and PyTorch's own timeout never fires.

It also explains the group-count dependence you report: each new communicator registers
new buffers, so more groups means more registration churn and more chances to hit a stale
entry. We saw 0/32 hangs with ~4 communicators and 4/5 with 8, at the same 4 nodes.

**Workaround** — `export FI_MR_CACHE_MONITOR=userfaultfd`. 0 hangs in 18 attempts across
4 and 8 nodes against a baseline of 21/23, and it costs nothing: uncapped all_reduce bus
bandwidth is 88.1 GB/s with it set vs 87.6 GB/s without at 8 nodes / world 64. `kdreg2`
works equally well (0/5) and may be cheaper than userfaultfd's page-fault path; we did
not compare registration overhead.

Note `NCCL_MAX_NCHANNELS=8` also stops it in this configuration but costs 21% of
training-band bandwidth, so the monitor is the better fix.

**One question for maintainers:** libfabric's own `fi_info -e` text says *"Userfaultfd is
the default if available on the system"*, and userfaultfd is compiled into this
container's libfabric. Yet selection lands on `memhooks`. If that defaulting were
corrected, this would stop reaching users without anyone setting a variable — which seems
worth more than documenting the workaround.

Environment matches yours: `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`,
PyTorch 2.10.0+rocm7.0, RCCL 2.26.6, ROCm 7.0.2, libfabric 2.1.0 in-container.
