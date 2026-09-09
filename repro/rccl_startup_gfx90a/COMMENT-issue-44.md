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


---

# Follow-up comment (why the default is memhooks, and what to change)

Ready to post as a follow-up.

---

Tracked down why the default lands on `memhooks`: **it is simply the documented default.**
HPE documents `memhooks` as the default MR cache monitor
(support.hpe.com, docId `dp00004854en_us`), and libfabric confirms it never even tries
anything else — with `FI_LOG_LEVEL=info FI_LOG_SUBSYS=mr` the unset case logs
`variable mr_cache_monitor=<not set>` and there is **no** `"Memory monitor uffd failed to
start"` warning anywhere, though that string is in the binary and `FI_WARN` is visible at
the default log level. So no probe fails; memhooks is just what is selected.

Worth noting libfabric's own `fi_info -e` text says *"Userfaultfd is the default if
available on the system"*, which contradicts HPE's documented default — that sent us down
a wrong path for a while and may be worth reporting separately.

**`FI_MR_CACHE_MONITOR=userfaultfd` is a genuine fix, not a cache bypass.** Verified by
inspecting the live process's own fd table after a collective:

| `FI_MR_CACHE_MONITOR` | `anon_inode:[userfaultfd]` fds open |
| --- | --- |
| unset (default) | 0 |
| `userfaultfd` | **1** |
| `memhooks` / `kdreg2` / `disabled` | 0 |

One wrinkle for anyone reproducing this: LUMI sets `vm.unprivileged_userfaultfd = 0`, so a
bare `userfaultfd()` returns EPERM and only succeeds with `UFFD_USER_MODE_ONLY`.
libfabric's monitor passes that flag, so it works — but a naive availability test will
suggest userfaultfd is unusable here when it is not.

**Suggested change:** set `FI_MR_CACHE_MONITOR` in the container's environment defaults
(`userfaultfd`, or `kdreg2` — both measured clean). The documented default is unsafe for
ROCm workloads, and fixing it in the image would retire this for every user rather than
relying on each of them finding the variable.
