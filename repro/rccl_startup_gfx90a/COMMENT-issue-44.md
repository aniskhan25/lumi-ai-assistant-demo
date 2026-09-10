# recipes#44 — what is posted, and one consolidating comment left to post

## Already posted (2026-09-09)

1. `FI_MR_CACHE_MONITOR` defaults to `memhooks` and that is the culprit; workaround is
   `export FI_MR_CACHE_MONITOR=userfaultfd`.
2. `fi_info -e` claims userfaultfd is the default "if available", which contradicts it.
3. The `vm.unprivileged_userfaultfd` story, struck through, with the correction that the
   default is `memhooks` per HPE docId `dp00004854en_us`.

## Draft: consolidating comment (adds what the thread does not yet have)

---

Some follow-up evidence, since the above was pieced together across three comments.

**Verified against your reproducer, unmodified.** 4 nodes / 32 ranks, your `timeout 180`
wrapper, 5 attempts per condition:

| condition | hung |
| --- | --- |
| default | **3/5** — stopped after 160 and 64 `collective done` lines, i.e. group 6 and group 3, matching your "which group it stalls on varies" |
| `FI_MR_CACHE_MONITOR=userfaultfd` | **0/5** — all 256 lines (32 ranks x 8 groups) every time |

**`kdreg2` works equally well** and is the other real monitor available here:

| monitor | hung | scale |
| --- | --- | --- |
| `userfaultfd` | 0/18 | 4 and 8 nodes |
| `kdreg2` | 0/13 | 4 and 8 nodes |
| `memhooks` (the default) | 4/5 | 4 nodes |
| `disabled` | 0/5 | 4 nodes |

**The fix is a real monitor, not a cache bypass.** Worth stating because `disabled` also
stops the hang, so "it stopped hanging" alone cannot distinguish the two. Inspecting the
live process's own fd table after a collective:

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
reproducer does, which is why your reproducer found this and ours initially did not.

**Suggested change:** set `FI_MR_CACHE_MONITOR` in the container's environment defaults
(`userfaultfd` or `kdreg2`). The documented default is unsafe for ROCm workloads, and
fixing it in the image retires this for every user instead of each one having to find the
variable. Happy to send a PR against the recipe if that is the preferred route.

Practical note for anyone testing: LUMI sets `vm.unprivileged_userfaultfd = 0`, so a bare
`userfaultfd()` returns EPERM and only succeeds with `UFFD_USER_MODE_ONLY`. libfabric's
monitor passes that flag, so it works — but a naive availability check will wrongly
suggest userfaultfd is unusable here.

---

## Not claimed, deliberately

- Not tested at 16 nodes, which the issue also reports.
- No vLLM-level confirmation: our vLLM runs never hung, so we have never watched the fix
  cure a hang in the real service, only in the probe and in your reproducer.
- Root cause of `memhooks` + ROCm (why allocator interception misses ROCm remapping) is
  not established — that belongs with libfabric or ROCm.
