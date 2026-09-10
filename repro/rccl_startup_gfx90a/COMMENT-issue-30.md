# Draft comment for lumi-ai-factory/laifs-container-recipes#30

The issue title says `NCCL_NET_GDR_LEVEL` "may cause jobs to hang". We can make that
precise — it is deterministic, and it has a clean scale boundary.

`torch.distributed` + RCCL, LAIF container `...20260807_115122`, stock settings otherwise,
`NCCL_NET_GDR_LEVEL=PHB` the only variable changed:

| scale | world size | baseline | `NCCL_NET_GDR_LEVEL=PHB` |
| --- | --- | --- | --- |
| 1 node | 8 | 0/3 | **0/3** |
| 2 nodes | 16 | 0/3 | **3/3 hung** |
| 4 nodes | 32 | 0/32 | **32/32 ranks hung** |
| 8 nodes | 64 | intermittent | **64/64 ranks hung** |

So it is harmless while all traffic is intra-node and hangs **every rank, every time**, as
soon as a collective crosses a node boundary. The hang is in the first cross-node
collective; `init_process_group` completes normally (0.7-1.4 s) in the same runs.

The 4-node row is the cleanest evidence: the background hang rate there is zero across 32
attempts, so there is no intermittency to confound it.

Two notes that may be useful:

- `NCCL_MAX_NCHANNELS=8` rescues it (0/5), and so does `NCCL_NET=Socket`, but
  `NCCL_RUNTIME_CONNECT=0` and `NCCL_SOCKET_IFNAME` do not — which places the failure in
  the OFI/CXI path rather than in RCCL's own logic.
- This is separate from #44. That one is the `memhooks` MR-cache-monitor hang and is
  intermittent and communicator-count dependent; this one is deterministic and scale
  dependent. Fixing `FI_MR_CACHE_MONITOR` does not help here.

Given the thread notes this originates in HPE's `ccl_env.sh`, the practical suggestion is
that single-node examples carrying `NCCL_NET_GDR_LEVEL=PHB` be flagged as unsafe to copy
into multi-node scripts — it is dormant at one node, which is exactly why it survives in
circulation until someone scales out.
