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

# Follow-up comment (why the default lands on memhooks)

Ready to post as a follow-up.

---

Found the reason libfabric falls back to `memhooks` here, and it points at a one-line
class of fix rather than a per-user workaround.

**LUMI sets `vm.unprivileged_userfaultfd = 0`.** On this kernel that does not forbid
userfaultfd — it requires the caller to pass `UFFD_USER_MODE_ONLY`. Measured directly in
the container as an unprivileged user:

| flags to `userfaultfd()` | result |
| --- | --- |
| `O_CLOEXEC` | EPERM |
| `O_CLOEXEC \| UFFD_USER_MODE_ONLY` | **OK** |

**libfabric's availability check for the default appears to omit that flag**, so it
concludes uffd is unavailable and falls back to `memhooks`. Confirmed from inside a live
process by inspecting its own fd table for `anon_inode:[userfaultfd]` after a collective:

| `FI_MR_CACHE_MONITOR` | uffd fds open |
| --- | --- |
| unset (default) | **0** |
| `userfaultfd` | **1** |
| `memhooks` / `kdreg2` / `disabled` | 0 |

So the explicit request takes the monitor's real open path, which does pass the flag, and
works — while the default silently degrades to `memhooks`, which is the monitor that
misses ROCm remapping.

Worth stressing: setting `FI_MR_CACHE_MONITOR=userfaultfd` is a **genuine fix, not a
cache bypass**. It opens a real userfaultfd and the MR cache stays active. (`kdreg2` also
works, 0/5, and is another option.)

**Suggested fix:** have the userfaultfd availability probe pass `UFFD_USER_MODE_ONLY`, or
set `FI_MR_CACHE_MONITOR` in the container's environment defaults. Either would retire
this for every LUMI user without anyone setting a variable — and the same mis-detection
will affect any hardened kernel with `vm.unprivileged_userfaultfd = 0`, not just LUMI.
