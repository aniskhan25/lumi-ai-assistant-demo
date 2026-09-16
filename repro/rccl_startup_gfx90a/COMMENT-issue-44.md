# recipes#44 — comment drafts

## Already posted (2026-09-09)

1. `FI_MR_CACHE_MONITOR` defaults to `memhooks` and that is the culprit; workaround is
   `export FI_MR_CACHE_MONITOR=userfaultfd`.
2. `fi_info -e` claims userfaultfd is the default "if available", which contradicts it.
3. The `vm.unprivileged_userfaultfd` story, struck through, with the correction that the
   default is `memhooks` per HPE docId `dp00004854en_us`.

Everything below is ready to paste. The framing is the corrected one: **`memhooks` is the
vendor-documented default and nothing mis-detects anything** — an earlier draft asked
maintainers to fix libfabric's userfaultfd availability probe, which was a mechanism we
invented to explain a measured outcome and never tested. There is no failed probe.

---

## Comment A — verification, alternatives, and cost

Some consolidated evidence, since the above was pieced together across three comments.

**Verified against your reproducer, unmodified.** 4 nodes / 32 ranks, your `timeout 180`
wrapper, 5 attempts per condition:

| condition | hung |
| --- | --- |
| default | **3/5** — stopped after 160 and 64 `collective done` lines, i.e. group 6 and group 3, matching your "which group it stalls on varies" |
| `FI_MR_CACHE_MONITOR=userfaultfd` | **0/5** — all 256 lines (32 ranks x 8 groups) every time |

**`kdreg2` works equally well**, and is the other real monitor available here:

| monitor | hung | scale |
| --- | --- | --- |
| `userfaultfd` | 0/18 | 4 and 8 nodes |
| `kdreg2` | 0/13 | 4 and 8 nodes |
| `memhooks` (the default) | 4/5 | 4 nodes |
| `disabled` | 0/5 | 4 nodes |

**The fix is a real monitor, not a cache bypass.** Worth stating because `disabled` also
stops the hang, so "it stopped hanging" cannot distinguish the two. Inspecting the live
process's own fd table after a collective:

| `FI_MR_CACHE_MONITOR` | `anon_inode:[userfaultfd]` fds open |
| --- | --- |
| unset (default) | 0 |
| `userfaultfd` | **1** |
| `memhooks` / `kdreg2` / `disabled` | 0 |

**It costs nothing.** Uncapped all_reduce bus bandwidth at 8 nodes / world 64: 88.1 GB/s
with the monitor set vs 87.6 GB/s without — identical within noise. For comparison
`NCCL_MAX_NCHANNELS=8`, which also stops the hang, costs 21% in the 128 MiB - 1 GiB band.

**On the group-count dependence you noted:** it is the governing variable, not rank count.
Same 4 nodes, same settings — 0 hangs in 32 attempts with ~4 communicators, 4/5 with 8.
Our own null result at 4 nodes was purely an artefact of building fewer groups than your
reproducer does, which is why yours found this and ours initially did not.

**Suggested change:** set `FI_MR_CACHE_MONITOR` in the container's environment defaults
(`userfaultfd` or `kdreg2`). The documented default is unsafe for ROCm workloads, and
fixing it in the image retires this for every user instead of each one having to find the
variable. Happy to send a PR against the recipe if that is the preferred route.

---

## Comment B — why the default is `memhooks`, and what it implies

Confirming the correction above with libfabric's own logging, in case it is useful.

With `FI_LOG_LEVEL=info FI_LOG_SUBSYS=mr`, the unset case logs
`variable mr_cache_monitor=<not set>` and then initialises monitors — and there is **no**
`"Memory monitor uffd failed to start"` warning anywhere, although that exact string is
in the binary and `FI_WARN` is visible at the default log level. So userfaultfd is never
attempted: `memhooks` is simply what the build selects, matching HPE's documented default.

That makes the mechanism straightforward. `memhooks` is the only monitor that detects
remapping by **intercepting userspace allocator calls**; `userfaultfd` and `kdreg2` both
observe mappings at the kernel level, and `disabled` removes the cache. All three of those
are clean and only the interception approach hangs — consistent with ROCm memory
operations not being reliably visible to allocator interception, leaving a stale
registration whose RDMA silently never completes. `disabled` being clean also says the
registration *cache* is the mechanism rather than the monitor choice being incidental.

One practical note for anyone reproducing this: LUMI sets
`vm.unprivileged_userfaultfd = 0`, so a bare `userfaultfd()` returns EPERM and only
succeeds with `UFFD_USER_MODE_ONLY`. libfabric's monitor passes that flag and works fine —
but a naive hand-written availability test will suggest userfaultfd is unusable here when
it is not. (We briefly concluded exactly that, wrongly.)

---

## Comment C — when does this actually bite?

One practical note on reachability, since it explains why some multi-node jobs never see
it. Communicator count matters more than scale — plain `torch.distributed`, stock
settings:

| nodes / world | communicators | hung |
| --- | --- | --- |
| 4 / 32 | ~4 | **0/32** |
| 4 / 32 | 8 | **4/5** |
| 8 / 64 | ~4 | 2/8 |
| 8 / 64 | 8 | 8/8 |

So a 4-node job building only a handful of groups can run clean indefinitely, which is why
our own production vLLM runs at that size never hit it, while the same nodes with your
8-group reproducer hang 4 times in 5. Useful for triage: "does it affect me?" depends
mainly on how many process groups the workload creates, not on node count. A plain DDP job
with one group is low risk; vLLM builds `tp`, `pp` and `ep` plus the world group, and
DeepSpeed or Megatron build more.

---

## Not claimed, deliberately

- Not tested at 16 nodes, which the issue also reports.
- Root cause of `memhooks` + ROCm — why allocator interception misses ROCm remapping — is
  not established; that belongs with libfabric or ROCm.
- No vLLM-level confirmation of a *cured hang*: our vLLM runs with the fix have all been
  healthy, but we never watched the fix turn a hanging vLLM run into a passing one.
