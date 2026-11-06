# DeepSeek-V4-Flash garbage output on LUMI-G (gfx90a): found and fixed

Branch `bug/dsv4-flash-gfx90a-garbage`, from `exp/deepseek-v41-gfx90a`. That branch's phase 0
got DeepSeek-V4-Flash serving on LAIF 20261028 (vLLM 0.26), but the output was garbage
(`'AskTreeLabel240uspaplenty成立 anthropž हज़ार…'`, finite logprobs). Lexivo reported the same
on vLLM 0.22.1.

## Result

**Root cause: vLLM's TileLang hyper-connection (mHC) "pre" kernel gives wrong output on
gfx90a.** vLLM 0.26 already keeps gfx942 off TileLang mHC ("TileLang MHC currently produces
incorrect results on gfx942", `vllm/model_executor/layers/mhc.py`), but enables it on every
other ROCm GPU, MI250X included. With the torch/Triton fallbacks that gfx942 uses,
**V4-Flash is correct on gfx90a**. Job 23494641, TP=8, one dev-g node:

| sanity | needle 2.1k / 16.8k tokens | GSM8K (200, greedy, thinking off) | garbage / non-finite |
|---|---|---|---|
| **5/5** (incl. *työsopimuslaki (55/2001)*) | **ok / ok** | **98.0 %** | 0 / 0 |

## How it was found

**1. Ground truth on the same hardware (job 23493492).** DeepSeek's own reference
implementation (`reference_upstream/`, verbatim from the checkpoint's `inference/`) needs
TileLang kernels written for NVIDIA. `reference_torch/` replaces its six kernels and the
CUDA Hadamard transform with plain-PyTorch versions of the same arithmetic:

- `act_quant`, FP8 1×128 with `ue8m0` scales;
- `fp4_act_quant`, e2m1 with round-to-nearest-even;
- `fp8_gemm` and `fp4_gemm`;
- `sparse_attn` with an attention sink;
- `hc_split_sinkhorn`;
- the Hadamard transform.

`selfcheck_kernels.py` checks them on CPU. Run at MP=8 on one node (`run_reference.sh`), the
reference answered "Helsinki", "391" and *"The contract is invalid if it has been made under
duress."* So the checkpoint, the FP8/FP4 formats, PyTorch on ROCm and RCCL are all fine, and
the bug is in vLLM's V4 ROCm path.

**2. Suspect.** vLLM's V4 decoder layer takes the fused TileLang mHC path whenever
`HAS_TILELANG_MHC`, which is true on gfx90a. The reference had used plain PyTorch for the
same maths.

**3. Switch (job 23494525).** `DSV4_TORCH_MHC=1` (`plugin/`) sets `HAS_TILELANG_MHC = False`,
so gfx90a runs the same torch/Triton fallbacks as gfx942. Every answer became correct. The
18k-token needle then OOMed in the indexer's torch fallback (`fp8_mqa_logits_torch`
materialises a dense fp32 `[heads, rows, kv]` score: 4 GiB per 8192-token prefill chunk).
The plugin now computes it in row chunks capped at 512 MiB, which is the same arithmetic per
row. With that, job 23494641 passed everything above.

**4. Which kernel (job 23509871).** `test_mhc_tilelang.py` compares each TileLang mHC op on
one GCD with vLLM's own fallback, using V4-Flash's real layer-5 and head parameters:

| TileLang op on gfx90a | result (T = 1, 7, 64, 300 tokens) |
|---|---|
| `mhc_pre`, mixing weights `post_mix`, `res_mix` (Sinkhorn) | correct, rel err ≤ 2e-5 |
| **`mhc_pre`, `layer_input`** (Σ pre·residual over the 4 streams) | **wrong, rel err 0.66–0.99** |
| `mhc_post` | correct (bit-exact at T=1) |
| `mhc_fused_post_pre`: residual, post_mix, res_mix | correct |
| **`mhc_fused_post_pre`, `layer_input`** (same pre stage) | **wrong, rel err 0.72–1.48** |
| `hc_head`, TileLang and Triton, vs DeepSeek's reference formula | both correct |

So every layer computed its mixing weights right and then fed itself a wrong input. The
defect is in the stage of the TileLang pre kernel that applies the pre weights to the
residual streams, both standalone and inside the fused post+pre kernel. It is not in Sinkhorn
or in the post/head kernels.

## Recipe: DeepSeek-V4-Flash on one LUMI-G node (LAIF 20261028, vLLM 0.26)

```bash
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 TORCH_MHC=1 \
  sbatch repro/deepseek_v41_gfx90a/run_v4_serve.sh     # from the repo root; TP=8, eager
```

Add `COMPILE=1 TRITON_LOGITS=1 NO_ASYNC=1` for piecewise CUDA graphs and the batched indexer
kernel (1.8–2.0× faster than eager, same accuracy; see Speed).
That is five changes against the stock container:

| | why |
|---|---|
| `--kv-cache-dtype fp8` | V4 stores K/V only as `fp8_ds_mla` |
| `--moe-backend triton_unfused` | MXFP4 experts; ROCm's automatic choice is aiter only |
| `AMDGCN_USE_BUFFER_OPS=0` | Triton 3.6's AMD `CanonicalizePointers` pass asserts compiling `_fused_q_kv_rmsnorm_kernel` |
| `DSV4_INDEXER_GATE=1` (plugin) | 0.26's sparse indexer is aiter-only; its op runs on Triton/torch fallbacks once the gate is lifted. Also chunks the torch MQA-logits fallback |
| `DSV4_TORCH_MHC=1` (plugin) | **this bug**: TileLang mHC pre-stage is wrong on gfx90a |

The plugin goes on `PYTHONPATH` through `singularity exec --env`, because the container
overrides the host `PYTHONPATH`. `run_v4_serve.sh` handles that.

## Speed (job 23510042, eager, TP=8, one dev-g node)

Streaming `vllm bench serve`, 1024 in / 512 out, same method as the Qwen3.8-Flash table
(`PROBE=0 BENCH=1`; fewer prompts per point because it is slow):

| concurrency | out tok/s | median TTFT | median TPOT | p99 TPOT | Qwen3.8-Flash eager / compiled TPOT |
|---|---|---|---|---|---|
| 1 | 3.0 | 0.7 s | 337 ms | 339 ms | 130 / 22 ms |
| 4 | 11.1 | 2.8 s | 358 ms | 361 ms | 132 / 32 ms |
| 16 | 37.2 | 8.1 s | 414 ms | 428 ms | 165 / 42 ms |
| 32 | 60.4 | 14.3 s | 503 ms | 526 ms | 168 / 47 ms |

**Correct but slow.** TPOT barely grows from 1 to 32 users (337 → 503 ms) while throughput
scales 20×. So each decode step is dominated by fixed per-step overhead, not compute. The
likely sources, all on fallbacks that only gfx90a takes:
- the torch mHC fallback: 43 layers × 2 Sinkhorn normalisations × 20 iterations of small ops
  per step;
- the indexer run as a Python op;
- eager launches throughout.

TTFT is long at load (8–14 s) because prefill goes through the torch MQA-logits fallback. The
KV cache holds 106k tokens (19 GiB per GCD), which is 3.25 concurrent requests at 32k context.

### With graphs (branch `perf/dsv4-flash-gfx90a-compile`, job 23513552)

V4 on vLLM 0.26 is not torch.compiled (compilation mode NONE). Without `--enforce-eager` it
captures "breakable" CUDA graphs: ops marked `eager_break_during_capture` (the V4 attention
and the sparse indexer) run eagerly between graph segments. Two changes were needed:

- the plugin registers `torch.ops.vllm.rocm_aiter_sparse_attn_indexer`, which 0.26 does only
  with aiter, and calls the indexer through it;
- `cudagraph_mode=PIECEWISE`. The default FULL_AND_PIECEWISE captures decode whole, indexer
  included, and the indexer's torch paged-MQA-logits fallback calls `.item()` per sequence,
  which cannot be captured (job 23512857: `hipErrorStreamCaptureUnsupported`).

`COMPILE=1` in `run_v4_serve.sh` does both. Accuracy is unchanged: sanity 5/5, needles
2.1k/16.8k ok, **GSM8K 98.0 %**, no garbage or non-finite logprobs.

| concurrency | out tok/s | median TTFT | median TPOT | p99 TPOT | vs eager (tok/s) |
|---|---|---|---|---|---|
| 1 | 5.0 | 0.7 s | 197 ms | 197 ms | 1.67× |
| 4 | 17.9 | 2.6 s | 218 ms | 221 ms | 1.61× |
| 16 | 51.2 | 7.9 s | 298 ms | 311 ms | 1.38× |
| 32 | 76.0 | 14.3 s | 392 ms | 418 ms | 1.26× |

Graphs take 1.6 GiB, so the KV cache drops to 97k tokens (2.97× at 32k). The gain is largest
at low load, where launch overhead dominated, and shrinks at 32 users. What remains runs
eagerly in every layer: the sparse attention and the indexer, whose decode fallback
(`fp8_paged_mqa_logits_torch`) loops over sequences in Python with a host sync per sequence.
Prefill (TTFT) is unchanged, as expected, since prefill was never graphed.

### Batched Triton indexer logits (job 23527632)

The indexer's decode fallback, `fp8_paged_mqa_logits_torch`, loops over the batch in Python
with an `.item()` per sequence. `plugin/lumi_dsv4_debug/paged_logits.py` replaces it with one
Triton kernel: grid fixed by batch size and `max_model_len`, sequence lengths read on the
device, so no host sync and capturable. `test_paged_logits.py` (job 23518750, one GCD):

- matches the torch fallback in 13 cases (block sizes 64/256, lengths 1 to 32k, batch up
  to 32): same -inf mask, rel err ~1e-7, identical top-512;
- captures in a CUDA graph and replays correctly with new lengths;
- per call at 32 sequences: torch 7.0 ms, Triton 0.05–0.2 ms. V4-Flash calls it in ~20
  layers per decode step, so the torch loop cost ~140 ms per step at 32 users.

**It needs `--no-async-scheduling`.** With async scheduling the server hangs after a few
decode steps of the first request (jobs 23518755, 23519175, 23527189): workers stop
answering, with no fault or exception. In job 23519175 the first 40 calls were checked
against the torch fallback (each check syncs) and were correct, and the hang came right
after the checks ran out. So it is a race that the fallback's per-sequence syncs hid, not
wrong logits. Not yet located (faulthandler stack dumps did not reach the log).

`TRITON_LOGITS=1 NO_ASYNC=1 COMPILE=1`, piecewise graphs. Accuracy: sanity 5/5, needles
2.1k/16.8k ok, **GSM8K 97.5 %** (vs 98.0 %: one question, within noise), 384 s vs 558 s.

| concurrency | out tok/s | median TTFT | median TPOT | p99 TPOT | vs piecewise above | vs eager |
|---|---|---|---|---|---|---|
| 1 | 5.3 | 0.7 s | 186 ms | 186 ms | 1.06× | 1.8× |
| 4 | 20.3 | 2.4 s | 192 ms | 195 ms | 1.13× | 1.8× |
| 16 | 71.0 | 7.6 s | 210 ms | 224 ms | 1.39× | 1.9× |
| 32 | 123.2 | 10.0 s | 240 ms | 257 ms | 1.62× | 2.0× |

TPOT now grows only 186 → 240 ms from 1 to 32 users, and the gain grows with load, as the
per-call timings predicted. With `--no-async-scheduling` vLLM also reports twice the KV
capacity from the same 17.45 GiB (190k tokens, 5.8× at 32k, vs 97k with async; jobs
23527079 and 23527632): observed, cause not investigated.

**FULL decode graphs** (`CUDAGRAPH_MODE=FULL_AND_PIECEWISE`, job 23529517) now capture, since
the indexer no longer syncs, and are correct (sanity 5/5, needles ok, GSM8K 97.0 %), but
give nothing over piecewise: 5.4 / 20.3 / 71.5 / 123.6 tok/s at 1/4/16/32 users (TPOT
185/193/209/239 ms), within 0.5 % of the table above, for 0.1 GiB more graph memory. So
with the indexer loop gone, the eager breaks around attention and the indexer are no longer
the cost; the GPU kernels are. Piecewise stays the recommendation.

### Where decode time goes (job 23530527)

`PROFILE=1` (eager, recipe switches, torch profiler on decode at 1 and 32 users; rank 0 split
by `analyze_trace.py`, which maps each GPU kernel to its Python call site):

| component | ms/step, 1 user | ms/step, 32 users | main kernel |
|---|---|---|---|
| **FP8 linear (W8A8 block)** | **89.0** | **94.0** | `_w8a8_triton_block_scaled_mm` (default config) |
| hyper-connections (torch fallback) | 54.0 | 60.0 | many small reduce/elementwise kernels |
| MoE experts (MXFP4, `triton_unfused`) | 17.7 | 18.8 | `_matmul_ogs_..._bf16xbf16xmxfp4` |
| sparse attention + other attention | 18.3 | 15.5 | `_sparse_attn_decode_ragged_kernel`, hipBLASLt |
| compressor, indexer, norms, head, other | 2.9 | 6.8 | |
| *sum (compute)* | *~182* | *~195* | |

All-reduce shows ~390 ms/step in this eager trace, but that is RCCL kernels spinning while
ranks wait for each other under the profiler's host overhead; the compute sum matches the
graph-mode TPOT (186 ms at 1 user), so communication is small there.

**The dense FP8 linears are half of every decode step** and do not grow with batch (89 →
94 ms), so they are latency-bound, not bandwidth-bound. Their weights are about 22 M FP8
parameters per layer per rank (shapes N×K 1536×4096, 4096×1024, 4096×256, 512×4096,
8192×1024): ~0.95 GB per step, which MI250X's HBM reads in about 1 ms. The block-scaled
Triton kernel runs on vLLM's default config (the startup log warns), with few programs at
small M, and MI250X has no FP8 units. The hyper-connection fallback is the second item (~30 %).

## Open

- **Speed**, in likely order of payoff:
  1. ~~Graphs~~: done, piecewise, 1.3–1.7× (above).
  2. ~~Batched indexer logits~~: done, 2.0× eager at 32 users (above). FULL decode graphs
     add nothing. Open: the race with async scheduling.
  3. **Dense FP8 linears (~50 % of decode, job 23530527)**: tuned W8A8 block configs for
     the five shapes, or dequantise to bf16 at load (exact: e4m3 values times power-of-two
     ue8m0 scales fit bf16) and use hipBLASLt / skinny GEMMs, for ~1 GB more per GCD.
  4. A correct hyper-connection pre kernel (~30 %).
  5. MXFP4 MoE (~10 %): `triton_kernels` picks its own tiles; no vLLM config to tune.
- **FP4 indexer cache:** vLLM's FP4 indexer-cache kernels crash silently on gfx90a
  (`../deepseek_v41_gfx90a`). V4-Flash uses the FP8 cache, so this run doesn't touch them.
- **To report** (laifs-container-recipes and upstream vLLM):
  1. TileLang mHC pre stage is wrong on gfx90a. Exclude all non-gfx950 ROCm, or at least
     gfx90a as gfx942 is, and attach `test_mhc_tilelang.py`.
  2. The Triton 3.6 `CanonicalizePointers` assertion.
  3. The aiter-only sparse-indexer gate (upstream main still opens it only for gfx950).
  4. The dense fp32 score in `fp8_mqa_logits_torch`.
- **DeepSeek-V4.1-Flash:** its blocker on gfx90a was that the shared V4 attention stack gave
  wrong output. That is now explained and worked around, so the stack is viable on gfx90a.
  The port itself is still large and the model needs two nodes (see
  `../deepseek_v41_gfx90a/README.md`).

## Files

| path | what |
|---|---|
| `reference_upstream/` | DeepSeek's `inference/` (model, generate, convert, config), verbatim |
| `reference_torch/` | plain-PyTorch `kernel.py` and `fast_hadamard_transform` stand-ins |
| `selfcheck_kernels.py` | CPU checks of `reference_torch/` |
| `run_reference.sh` | convert to MP=8 (once, ~24 min) and run the reference greedily |
| `plugin/` | vLLM 0.26 general plugin: `DSV4_INDEXER_GATE`, `DSV4_TORCH_MHC`, `DSV4_TRITON_PAGED_LOGITS` |
| `test_mhc_tilelang.py`, `run_mhc_test.sh` | per-kernel TileLang mHC check on one GCD |
| `plugin/lumi_dsv4_debug/paged_logits.py` | batched Triton indexer decode logits (`DSV4_TRITON_PAGED_LOGITS`) |
| `test_paged_logits.py`, `run_paged_logits_test.sh` | that kernel vs the torch fallback, graph replay, timing |
| `../deepseek_v41_gfx90a/run_v4_serve.sh` | the serving job (switches documented at its top) |
