# Model survey for Aitta on LUMI-G

Per-user serving speed for the models a customer moving up from GLM-4.5-Air would ask
about, measured the same way as `repro/kimi_k27_aitta/`: streaming `vllm bench serve`,
concurrency 1/4/16/32, TTFT and TPOT p50/p99, warm-up first, one model per allocation.
Two workloads per server: `8192:512` (coding-like) and `1024:1024` (input ~ output, as in
the colleague's sheet).

| run | model | nodes | layout | notes |
|---|---|---:|---|---|
| qwen | `Qwen/Qwen3-Coder-480B-A35B-Instruct` (BF16, 894 GiB) | 4 | TP=8 PP=4 | |
| r1 | `deepseek-ai/DeepSeek-R1-0528` (FP8, 642 GiB) | 2 | TP=8 PP=2, EP | |
| glm | `zai-org/GLM-4.7` (BF16, 667 GiB) | 2 | TP=8 PP=2, EP | substitute for GLM-5.x |
| kimi | `moonshotai/Kimi-K2.7-Code` (INT4, 554 GiB) | 3 | TP=8 PP=3, EP | Kimi B with both workloads |
| r1int4 | `RedHatAI/DeepSeek-R1-0528-quantized.w4a16` (INT4, 346 GiB) | 1 | TP=8, EP | r1 without FP8 |

```bash
sbatch repro/download_model.sh <model-id>                    # if not cached
DEPENDENCY=<download-jobid> bash repro/aitta_model_survey/submit.sh qwen
```

## Not runnable on LUMI-G: DeepSeek-V4, DeepSeek-V3.2, GLM-5.x

These use DeepSeek-style sparse attention (`index_topk` in the config). vLLM's only ROCm
backend for it is `ROCM_AITER_MLA_SPARSE`, which imports `aiter` (`get_mla_metadata_v1`,
`mla_decode_fwd`) unconditionally. Neither LAIFS vLLM container ships aiter (checked
2026-10-05: vLLM 0.22.1 of 20260807 and vLLM 0.26.0 of 20260929), and aiter supports only
gfx942/gfx950, not the MI250X's gfx90a —
[laifs-container-recipes#8](https://github.com/lumi-ai-factory/laifs-container-recipes/issues/8),
open. Their architectures are in vLLM's registry, so they fail at startup, not at
registry lookup.

## Results (2026-10-05)

TPOT median/p99 in ms, TTFT p99 in s. Every request completed unless noted.
Kimi-K2.7-Code B (3 nodes, `TP=8 PP=3`, job 22344021) from `repro/kimi_k27_aitta/` for
comparison; it ran 8192:512 only.

| model | nodes | startup | workload | 1 user | 4 users | 16 users | 32 users |
|---|---:|---:|---|---|---|---|---|
| Qwen3-Coder-480B (job 22550268) | 4 | 26 min | 8192:512 | 24/24, 0.19 s | 42/54, 7.9 s | 119/137, 37 s | 110/227, 69 s |
| | | | 1024:1024 | **23/23**, 0.11 s | 40/40, 2.9 s | 67/69, 0.46 s | 95/97, 2.4 s |
| GLM-4.7 (job 22550575) | 2 | **6 min** | 8192:512 | 42/42, 0.25 s | 67/84, 11.7 s | 168/188, 53 s | 139/309, 96 s |
| | | | 1024:1024 | 40/40, 1.5 s | 65/66, 1.2 s | 89/91, 4.2 s | 116/116, 1.1 s |
| Kimi-K2.7-Code B | 3 | 19 min | 8192:512 | 51/51, 0.38 s | 86/109, 16.9 s | 267/283, 63 s | 286/485, 123 s |
| DeepSeek-R1-0528 (job 22550269) | 2 | 17 min | 8192:512 | 312/312, 0.92 s | 358/438, 57 s | *hung* | — |
| Kimi-K2.7-Code (job 22572310) | 3 | 17 min | 8192:512 | 51/51, 0.40 s | 86/107, 14.7 s | 267/283, 64 s | 280/488, 126 s |
| | | | 1024:1024 | 49/49, 0.19 s | 81/82, 0.30 s | 150/152, 2.0 s | 226/226, 1.5 s |
| DeepSeek-R1-0528 INT4 w4a16 (job 22572325) | **1** | 7 min | 8192:512 | 54/54, 0.36 s | 89/113, 19.9 s | 294/308, 75 s | 452/542, 277 s |
| | | | 1024:1024 | 52/52, 0.57 s | 83/84, 2.1 s | 151/156, 8.2 s | 237/247, 16 s |

KV room for 65,536-token requests: Qwen 60x, Kimi 23x, GLM-4.7 11.5x, R1 INT4 3.3x
(one node, so 32 users with 8k prompts queue for KV: TTFT p99 277 s).

- **Qwen3-Coder-480B reproduces the colleague's single-request 23 ms** at 1024:1024, so
  the harnesses agree. Their 1-user TPOT and ours match; the per-user gap seen with Kimi
  is workload, not setup.
- **Prompt length is most of the multi-user penalty.** At 16 users, 8192-token prompts
  roughly double TPOT and add 1.1–1.6 s ITL spikes (new prefills inside decode steps)
  against 1024:1024. With short prompts Qwen and GLM-4.7 stay under 100 ms TPOT
  at 16 users.
- **DeepSeek-R1-0528 FP8 is ~7x slower per token than Kimi-K2.7-Code INT4**, the same
  DeepSeek-V3 architecture: 312 ms vs 46–51 ms at 1 user. Its FP8 layers use vLLM's
  default W8A8 block-FP8 kernel config ("might be sub-optimal"); gfx90a has no FP8
  hardware. This is the clearest evidence that FP8 checkpoints are the wrong choice on
  LUMI-G.
- **R1 hung at 16 users**, like Kimi A did. This time the NCCL watchdog recorded it: one
  GPU on node 2 (rank 11) waited 600 s on a pipeline `RECV` of 14336 elements from
  node 1, its node-2 peers blocked in an all-gather on it, node 1 in a 16-element
  broadcast. A cross-node point-to-point transfer silently never completed, with
  `FI_MR_CACHE_MONITOR=userfaultfd` set. So the stall is not specific to TP across nodes;
  PP across nodes hits it too. It hit 2 of 7 multi-node serving runs, both in the
  16-user phase; Qwen (4 nodes), GLM-4.7, Kimi B and D ran 16 and 32 users cleanly.
  Cancelled by hand: vLLM's engine timeout did not fire, so the client would have waited
  out the walltime.
- **Kimi reproduces across allocations:** job 22572310 matches Kimi B (job 22344021) to
  within 2% at every 8192:512 point, a day apart.
- **INT4 removes the FP8 penalty for DeepSeek-R1-0528:** 52–54 ms TPOT at 1 user against
  312 ms for the FP8 checkpoint, about 6x faster, on one node instead of two. Per user it
  now tracks Kimi-K2.7-Code (same architecture) within ~5%. One node also means no
  cross-node collectives, so no exposure to the multi-node stall.
