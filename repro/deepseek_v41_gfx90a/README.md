# DeepSeek-V4.1-Flash on LUMI-G (MI250X, gfx90a): phase 0

Should we try to port `deepseek-ai/DeepSeek-V4.1-Flash` to LAIF's vLLM 0.26, as we did for
Qwen3.8-Flash (branch `exp/qwen38-flash-gfx90a`)? This is the de-risking pass, before any
port work and before downloading its 475 GiB.

> **Status (2026-11-12): V4.1-Flash serves correctly and at V4's speed on one node: 14.0 /
> 264 / 400 tok/s at 1 / 32 / 64 users with `--max-num-seqs 64`, GSM8K 98.5 % (200), both
> needles, decode = prefill to 19.7k tokens (job 23668248). See "Phase 4: speed" below.
> Async scheduling (drop `NO_ASYNC=1`) is correct too and gives 15.2 / 282 / 428 tok/s, at
> the cost of ~43 % less KV cache (23 instead of 41 concurrent 32k requests, job 23696171).**
>
> **Status (2026-11-11): V4.1-Flash serves correctly on one node through a plugin on vLLM
> 0.26: GSM8K 99/100, no garbage, needles at 2.1k and 16.8k tokens right, decode agrees with
> prefill to 0.02–0.05 nats up to 20k tokens (job 23644138). See "Phases 2–3" below.**
>
> **Status (2026-11-10): the V4-Flash garbage is fixed; the V4.1 port blockers remain.**
> Branches `bug/dsv4-flash-gfx90a-garbage` and `perf/dsv4-flash-gfx90a-compile`
> (`repro/dsv4_flash_debug/README.md`) found two bugs:
> - vLLM's TileLang mHC "pre" kernel is wrong on gfx90a. It is replaced with Triton (`MHC_FIX=triton`).
> - Under CUDA graphs, NaN in pad rows reaches the MoE router and corrupts real rows. It is
>   guarded (`ROUTER_GUARD=1`).
>
> V4-Flash is now correct on one node: GSM8K 97.4–97.8 % with no garbage, and 298 / 437 tok/s
> at 32 / 64 users (jobs 23591543/23591544). Blockers 2 and 3 below are still bypassed, not
> fixed (`AMDGCN_USE_BUFFER_OPS=0`, `INDEXER_GATE=1`). So verdict point 1 no longer holds.
> Points 2 and 3 still do: the ~10k-line core sparse-MLA port, onto which the V4 fixes would
> have to be re-applied, and 475 GiB needing two nodes without PP.

## The model (config of 2026-09-10)

`DeepseekV41ForCausalLM`, 40 layers, hidden 5120, 384 routed experts with 6 active plus 1
shared.

- **Attention:** MLA with one KV head of width 512, of which 64 are rope dimensions. It is
  compressed sparse attention: per-layer compress ratios 2/1/0, a 128-token sliding window,
  and an indexer with 32 heads that keeps the top 512. On top of DeepSeek-V4 it adds
  cross-layer K/V sharing (`kv_source_layer_ids`) and candidate-block preselection.
- **Other:** 4-way hyper-connections with Sinkhorn normalisation, and Engram n-gram tables at
  layers 1 and 14 (the same idea as Qwen3.8's PLE, and upstream shares code between them).
  There are also DSpark/MTP draft layers.
- **Number formats:** FP8 weights in 32×32 blocks with `ue8m0` scales, and FP4 (MXFP4)
  experts. gfx90a has no FP8 or FP4 units, so both would be dequantised in software.
- **Size: 475 GiB. That does not fit in one LUMI-G node** (8 × 64 GB HBM), so it needs at
  least two nodes (TP=16 or PP=2), and the plugin approach has not ported PP.

## Static findings (vLLM main @ 790be38 vs LAIF's vLLM 0.26.0)

**Not in 0.26.** `DeepseekV41ForCausalLM` is absent; 0.26 has DeepSeek-V4 and V3.2. Upstream
has `vllm/models/deepseek_v41/{amd,nvidia,common}`, laid out like Qwen3.8's package.

**aiter on the default AMD path.** The ROCm attention (`amd/rocm.py`, class
`DeepseekV41ROCMAiterMLAAttention`) branches on `_ON_GFX950` and `rocm_aiter_ops.is_enabled()`,
and it has explicit non-gfx950, non-aiter branches. The ops it uses have Triton or torch
fallbacks:
- sparse prefill: Triton after the aiter variants;
- sparse decode: Triton, with a "fallback decode kernel" for architectures other than
  gfx942/gfx950;
- indexer logits: Triton and torch.

The MXFP4 indexer (`rocm_paged_mxfp4_indexer`) is "gfx950 only" and aiter-based, so gfx90a
must take the FP8/BF16 indexer. The optional monokernel decode (`amd/mono`, 9k lines) imports
aiter FlyDSL directly, but it is opt-in (`--monokernel`, gfx942/gfx950). So aiter is not a
hard wall, but nothing upstream tests the non-gfx94x/95x branches.

**The port would be much larger than Qwen3.8's:**

| | Qwen3.8-Flash | DeepSeek-V4.1-Flash |
|---|---|---|
| Upstream model code to carry (no NVIDIA, no monokernel) | ~9.4k lines | ~15k lines |
| Modules missing from 0.26 | 4 | **17** |
| Names missing from existing 0.26 modules | ~15 | **~48** |
| Core vLLM files whose newer version is needed | none | **the sparse-MLA stack, shared with 0.26's own V4** |

The core files have roughly doubled since 0.26, and V4.1 imports new names from them, so
their newer versions would have to replace 0.26's:

| file | 0.26 | main | lines changed |
|---|---|---|---|
| `v1/attention/ops/rocm_aiter_mla_sparse.py` | 2,400 | 4,986 | 3,018 |
| `v1/attention/backends/mla/indexer.py` | 960 | 2,053 | 1,721 |
| `model_executor/layers/sparse_attn_indexer.py` | 849 | 1,692 | 1,117 |
| `v1/attention/backends/mla/sparse_swa.py` | 782 | 1,393 | 945 |
| `models/deepseek_v4/attention.py` | 807 | 1,185 | 672 |
| `model_executor/layers/mhc.py` | 568 | 1,020 | 498 |
| `v1/attention/backend.py` | 1,110 | 1,160 | 486 |
| `compressor_utils.py`, plus 3 files new since 0.26 | | | ~1,700 |

That is roughly 10k changed lines in core attention code, against Qwen3.8's ~735 lines of
bridging and no core replacements.

## Runtime checks (2026-11-05, LAIF 20261028, vLLM 0.26.0+lumi_aif_gfx90a_568afb3)

### DeepSeek-V4-Flash end to end: four blockers, and then garbage output

DeepSeek-V4 is natively supported by 0.26, and V4.1 builds on its sparse-MLA attention,
indexer and compressor. V4-Flash is 149 GiB and fits one node, so it ran at TP=8 on one dev-g
node with `--moe-backend triton_unfused --enforce-eager` (`run_v4_serve.sh`). Each attempt got
one step further:

| job | setup | outcome |
|---|---|---|
| 23491881 | stock | `fp8_ds_mla layout only supports fp8 kv-cache, got auto`. A config requirement → `--kv-cache-dtype fp8` |
| 23491948 | + fp8 KV | Weights load; the first forward pass fails because **Triton 3.6 cannot compile V4's `_fused_q_kv_rmsnorm_kernel`**: an assertion in the AMD pass `TritonAMDGPUCanonicalizePointers` (`ConvertSCFIfOp`), triggered by a pointer chosen inside an `if`. The pass is AMD-generic, not gfx90a-specific, so V4 cannot run on ROCm with this container's Triton at all |
| 23492281 | + `AMDGCN_USE_BUFFER_OPS=0` (skips that pass) | `Sparse attention indexer ROCm path is only supported on AITER`. **0.26's indexer is gated aiter-only, and upstream main still is**, except gfx950. Inside the op every step has a non-aiter path: Triton cache insert and gather, compiled top-k, torch FP8 logits |
| 23492563, 23492716 | + `INDEXER_GATE=1` (`indexer_gate/`, calls the op's Python implementation directly) | **Serves (ready in 281 s, weights 104 s), but the output is garbage**: `'AskTreeLabel240uspaplenty成立 anthropž हज़ार हज़ार…'`, with finite logprobs. This matches Lexivo's report of broken V4-Flash output on 0.22.1 |

### vLLM 0.26's own DSV4 ROCm kernel tests on one GCD

Unmodified tests against the matching container (`run_kernel_tests.sh`; buffer ops on
23492901, off 23492965, same results):

| test file | result |
|---|---|
| `test_rocm_triton_attn_dsv4`: sparse prefill/decode, gfx90a fallback kernels | **19 passed**, 21 skipped (split-K decode is gfx942/950 only) |
| `test_fused_deepseek_v4_qnorm_rope_kv_insert` | **155 passed** |
| `test_compressor_kv_cache` | FP8 roundtrips and inserts pass; **the process dies at the first FP4 (`use_fp4=True`) case** |
| `test_fused_indexer_q_rope_quant` | FP8 cases pass; **the process dies at the first FP4 case** |
| `test_fused_inv_rope_fp8_quant` | 50/50 "fail" on a harness error (`missing ... 'launch_pdl'`): the test calls the Triton kernel without the PDL argument the ROCm build expects. **Inconclusive**, not a numerical failure |

The FP4 crashes are silent (exit 1, no traceback). They are not on the served path, which uses
the FP8 indexer cache (`use_fp4_indexer_cache=False`). Every tested component on the served
path passes. **So the garbage comes from something these tests do not cover.** Candidates:
- the torch fallback for indexer logits and top-k, reached only through the lifted gate;
- inverse rope with the FP8 `wo_a` path (its test could not run);
- MXFP4 expert dequantisation for V4's weight layout;
- the mHC ops;
- the FP8 KV encoding on gfx90a (the tests note OCP e4m3fn on gfx90a vs FNUZ on gfx942).

## Verdict

**Do not start a V4.1 port now.**

1. **V4.1 sits on top of DeepSeek-V4's attention stack, and that stack does not produce
   correct output on gfx90a today**, even for V4, which 0.26 supports natively. It took a
   compiler workaround and a gate bypass just to start, and then the output was garbage. A
   V4.1 port would inherit all of that. *(Superseded 2026-11-10: V4-Flash is now correct; see
   the status note at the top.)*
2. **The port itself is about an order of magnitude bigger than Qwen3.8's.** It means carrying
   newer copies of ~10k lines of core sparse-MLA attention code that 0.26's own V4 also uses,
   plus 17 missing modules.
3. **V4.1-Flash does not fit one node** (475 GiB), and our plugin approach has not ported
   pipeline parallelism.

**If the DeepSeek-V4 family matters on LUMI-G, the prerequisite is a correct V4-Flash on
gfx90a.** That is a debugging project in its own right: bisect the garbage by comparing
layer outputs, swapping the MoE backend, and checking the indexer fallback and FP8
encodings. Worth reporting meanwhile, to laifs-container-recipes and upstream:
- the Triton 3.6 `CanonicalizePointers` assertion;
- the aiter-only indexer gate, which blocks every non-gfx950 GPU without aiter;
- the FP4 indexer-cache crashes;
- the `launch_pdl` test break.

## Port plan, phases 0–1 (2026-11-10)

### Phase 0: checks without GPUs

- **No newer container.** `lumi-multitorch-latest.sif` is still 20261028 (vLLM 0.26), so V4.1
  needs a port.
- **PP exists for V4 in 0.26.** `DeepseekV4ForCausalLM` implements `SupportsPP`, using
  `make_layers`, `PPMissingLayer` and intermediate tensors.
- **The dense linears are MXFP8, not block-FP8.** V4.1's `quantization_config` has
  `weight_block_size [32, 32]` with `ue8m0` scales, while V4-Flash has `[128, 128]`. Upstream
  routes such layers to `ModelOptLinearMethod(QuantSpec(kMxfp8Static, kMxfp8Dynamic),
  CkptCtx(scale_block_size=(32, 32)))`, which 0.26 lacks.
  - 0.26's `ModelOptMxFp8LinearMethod` expects 1×32 weight scales, so the 32×32 checkpoint
    scales must be expanded at load time.
  - Its native MXFP8 kernel is gfx950-only (`supports_mx()`). On gfx90a only `emulation` is
    left.
  - The V4 fast path (`BF16_LINEAR`: dequantise to bf16 once, keep FP8 activation rounding)
    carries over, but the rounding becomes MXFP8: an e8m0 scale per 32 values, instead of
    V4's fp32 scale per 128.
  - Lesson from V4: dropping the activation rounding costs GSM8K points.
- **Import audit against current main** (`65711a1`; `amd/` + `common/`, no NVIDIA or
  monokernel; 32 files, 14.3k lines). 19 modules are missing from 0.26 (several are imported
  only in NVIDIA or optional branches) and 43 names are missing from existing modules. This
  is the same scale as at `790be38`.
- **Storage.** Scratch has 61 TiB free. `/flash` has 2 TiB free.

### Phase 1: V4-Flash on two nodes, PP=2 × TP=8 (job 23602435)

`run_v4_serve.sh` now has a multi-node mode: `sbatch --nodes=2` starts one vLLM per node via
srun, with mp, `--nnodes`/`--headless` and `FI_MR_CACHE_MONITOR=userfaultfd`. It used the
1-node recipe unchanged, with `--max-num-seqs 64`.

| | GSM8K (500) | sanity / needles | out tok/s at 1 / 4 / 16 / 32 / 64 users | TPOT at 1 / 32 / 64 |
|---|---|---|---|---|
| 1 node, TP=8 (23591544) | 97.8 % | 5/5, 2/2 | 15.5 / 56.2 / 187.9 / 298.0 / 436.6 | 64 / 97 / 136 ms |
| **2 nodes, PP=2 × TP=8** | **97.6 %**, 0 garbage | 5/5, 2/2 | 15.0 / 54.5 / 185.7 / 329.9 / 506.1 | 66 / 88 / 114 ms |

- **Accuracy.** The plugin fixes work unchanged across a pipeline stage boundary. The 0.2-point
  gap is one question.
- **KV cache.** Each stage gets about 310k tokens.
- **Throughput.** PP=2 is +11 % at 32 users and +16 % at 64, and −3 % at 1 user. That is per
  2 nodes, so per node it is lower; the point is that V4.1 needs two nodes anyway.
- **Loading weights took 1492 s, against ~104 s on one node.** 16 ranks read 149 GiB from
  Lustre at once. At V4.1's 475 GiB, staging the checkpoint on `/flash` is worth measuring
  before phase 3.
- **Exit status.** Slurm reports the job `FAILED` (exit 9). The step is SIGKILLed 32 s after the
  exit trap's SIGTERM, which is cosmetic. The probe passed and every sweep point ran.

## Phases 2–3: V4.1 served by a plugin on vLLM 0.26 (2026-11-11)

No newer container is coming, so V4.1 runs as a plugin (`lumi_dsv41/`) on LAIF 20261028:
upstream's V4.1 model and its sparse-attention stack are vendored at 65711a1
(`vendor_sync.py`) and adapted to 0.26 through import hooks, grafts and source patches
(`lumi_dsv41/hooks.py`). One node, TP=8 with expert parallelism, eager, MRV2, fp8 KV.

**Result (job 23644138): GSM8K 99/100, no garbage, all five sanity answers right, needles at
2.1k and 16.8k tokens right, and decode agrees with prefill to a mean of 0.02–0.05 nats for
prompts of 42 to 19.7k tokens.**

What it took on the real checkpoint, in order:

| Symptom | Cause | Fix |
|---|---|---|
| 1 h readiness timeout at shard 39/48 | 8 ranks each read 475 GiB from scratch; the page cache cannot hold it next to 189 GiB of pinned Engram tables | `stage_flash.sh` copies it to LUMI-F; load 724 s, then ~300 s once cached |
| OOM after load: 62 GiB of weights per GCD | 0.26's ROCm MXFP4 sizing rounds each rank's expert width (2304/8 = 288) up to 512 | `--enable-expert-parallel`: whole experts, 2304 = 9 × 256, no padding; 37 GiB per GCD |
| OOM in profiling: 32 GiB indexer workspace | upstream bounds decode-logit rows by `max_num_seqs` via `get_current_vllm_config()`, which 0.26 does not provide in the forward | capture the serving limits at model init (`_decode_rows_bound`) |
| Fluent but drifting text, GSM8K 3 %, 36/200 garbage | upstream bug: on GPUs other than gfx942/950 the fallback decode kernel skips the inverse RoPE yet reports it applied, so the caller skips its own pass | withhold the positions there (`_decode_inv_rope`); the caller's inverse RoPE then covers decode rows |
| Short prompts fine (GSM8K 100/100), but past ~3k tokens decode diverges from prefill from the 5th token; 16.8k needle answers `482` | upstream bug: on ROCm the indexer K cache is written in 16×16 tiles (`indexer_k_store.py`, SHUFFLE); prefill gathers it tile-aware, but without aiter the decode logits fall back to `fp8_paged_mqa_logits_torch`, which reads it row-major. Wrong scores pick the wrong top-512 once a context has more than 512 compressed positions | un-tile each gathered page in the fallback (`_paged_logits_unshuffle`) |

How the last one was found without a reference implementation (`decode_vs_prefill.py`,
`DIAG=1`): greedy decode records each token's logprob, then the same tokens are scored as a
prompt (prefill). They differed by up to 23 nats (mean 0.64–1.07) before the RoPE fix, and by a
mean of 0.01–0.05 after it for short prompts. Prompts of 3.3k–19.7k tokens still differed (mean
0.9–1.6, first >0.5 nats at the 5th token) until the indexer fix, and agree to 0.02–0.03 after. Ruled out on the way: Engram hashing (real hashes, correct
lookback ids), the SWA KV record (0.26's C++ encoder and upstream's OCP writer are
byte-identical on gfx90a, `test_kv_format.py`) and the compressor ring.

Memory per GCD: 37.2 GiB weights, 17.3 GiB KV cache (744k tokens, 22.7 concurrent 32k
requests). Host: 189 GiB of pinned Engram tables per node.

## Phase 4: speed (2026-11-12)

One node, 1024 tokens in, 512 out, `--max-num-seqs 32`; each row one allocation, and each
row after the first also passed the decode-vs-prefill check to 19.7k tokens, both needles and
GSM8K (99–100/100).

| setup | job | tok/s at 1 / 4 / 16 / 32 users | TPOT at 1 / 32 users |
|---|---|---|---|
| eager | 23653727 | 3.5 / 13.6 / 45.8 / 81.7 | 284 / 367 ms |
| + batched Triton indexer logits (`TRITON_LOGITS=1`) | 23658965 | 3.6 / 14.1 / 51.1 / 97.0 | 278 / 306 ms |
| + CUDA graphs (`COMPILE=1`, `NO_ASYNC=1`) | 23666441 | 9.4 / 35.9 / 117.7 / 183.7 | 106 / 157 ms |
| + mHC in Triton (`MHC_TRITON=1`) | 23667236 | 13.5 / 52.6 / 169.0 / 260.9 | 74 / 110 ms |
| + split-K prenorm GEMM in Triton, `--max-num-seqs 64` | 23668248 | **14.0 / – / – / 263.9**, **400.0 at 64** | **71 / 108 ms**, 146 at 64 |
| same, async scheduling (no `NO_ASYNC=1`) | 23696171 | 15.2 / – / – / 282.0, 427.8 at 64 | 65 / 95 ms, 130 at 64 |
| + custom all-reduce (`CUSTOM_AR=1`), `NO_ASYNC=1` | 23698376 | **14.8 / – / – / 279.8**, **423.5 at 64** | **67 / 102 ms**, 138 at 64 |

V4-Flash's recipe on the same node gives 15.0 / 53.8 / 177.1 / 273.3 (`../dsv4_flash_debug`).

- **Eager is launch-bound.** A decode step is ~14,600 kernel launches. In the eager profile
  (job 23660212, `PROFILE=1`) every rank spends ~80 % of its GPU time in the all-reduce
  kernel waiting for the other ranks' hosts to catch up. GPU work proper: mHC 47 ms, sparse
  attention 20, MoE 18, the rest under 10 (profiler-inflated).
- **Indexer logits** (`lumi_dsv41/paged_logits.py`): one batched launch, no host sync,
  reading the tiled cache. `test_paged_logits.py` (job 23658925): same -inf mask, rel. error
  ~1e-7 and the same top-512 as vLLM's torch fallback on the untiled copy, also under graph
  replay; 0.05–0.4 ms instead of 7 ms at 32 sequences.
- **CUDA graphs were silently off.** 0.26 enables its breakable CUDA graphs (PIECEWISE
  without torch.compile) automatically only for its own model names; for the plugin model
  it logs "torch.compile is turned on, but the model ... does not support it" and runs eager
  (job 23653728: no change). `COMPILE=1` now sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. The first
  graph run then died at 15 concurrent requests (job 23666313): 0.26's runner pads the
  token count to the graph size but not the request count, so upstream's indexer stored 15
  unpacked top-k rows into a 16-row slice. `_indexer_graph_pad` stores the real rows and
  marks pad tokens -1.
- **mHC** (`lumi_dsv41/mhc_triton.py`): with TileLang off on gfx90a and no aiter, V4.1's
  delayed pre and post ran as upstream's torch references, ~40 small kernels per call and
  ~80 calls per step. Now the fp32 prenorm GEMM in torch, then one Triton kernel for the
  gates, Sinkhorn and the collapse with the carried pre-mix (V4's big-fuse kernel, delayed),
  and one for the post. `test_mhc_triton.py` (job 23667193): gates within 1e-7, layer input
  within 1e-5 of the references, 1.12 → 0.18 ms per pre and 0.14 → 0.04 ms per post.
  For up to 64 tokens the prenorm GEMM and squared sum are one split-K Triton kernel too
  (job 23667826: gates within 3e-6, 0.18 → 0.10 ms per pre).
- **64 sequences** (job 23668248): GSM8K 98.5 % of 200, no garbage, both needles; V4's
  pad-row bug at 64 does not show (ROUTER_GUARD=1 is on by default). With
  `--no-async-scheduling` vLLM reports 1.35M tokens of KV cache (41 concurrent 32k requests).
- **Async scheduling** (job 23696171, the recipe without `NO_ASYNC=1`, same code): no hang
  in a 21-minute probe + sweep, GSM8K 98.0 % of 200 (4 misses vs 3: 2 shared, 2 only
  with async, 1 only without; noise), both needles, decode = prefill to 19.7k (the first three diag prompts
  identical to the last digit). **+8.5 % at 1 user, +6.9 % at 32, +7.0 % at 64** (TPOT
  71 → 65, 108 → 95, 146 → 130 ms): the host's batch preparation now overlaps the GPU step.
  The cost, as on V4: the KV cache estimate falls from 1.35M to 0.77M tokens (41 → 23
  concurrent 32k requests), because each sliding-window layer reserves its window plus the
  tokens of every batch in flight, and async keeps two batches in flight. Choose by
  workload: async for short prompts at up to 64 users, `NO_ASYNC=1` when many long
  contexts must fit at once.
- **Custom all-reduce** (job 23698376, `CUSTOM_AR=1` from the V4 plugin's `custom_ar.py`,
  merged from `perf/dsv4-flash-gfx90a-compile`): vLLM's one-shot all-reduce instead of RCCL
  for the TP group, with the input always copied into the uncached IPC buffer, which keeps it
  coherent across the GCDs' L2 caches (the stock kernel read stale peer data on V4, GSM8K 0 %).
  V4.1's AMD model keeps sequence parallelism off, so attention and MoE outputs are TP
  all-reduces, and the log shows `['CUSTOM', 'PYNCCL']` for `tp:0` (0.26 enables it for TP
  groups only). GSM8K 98.5 % of 200 (3 misses, 2 shared with the baseline), both needles,
  decode = prefill to 19.7k, KV unchanged (1.35M tokens). **+5.5 % at 1 user, +6.0 % at 32,
  +5.9 % at 64.**

Open: FULL decode graphs (no gain on V4); reporting the two upstream gfx90a
bugs (decode inverse RoPE, SHUFFLE indexer read) and the graph-padding mismatch.

## Reproduce

```bash
# from the repository root on LUMI
sbatch --partition=dev-g repro/deepseek_v41_gfx90a/run_kernel_tests.sh
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 sbatch repro/deepseek_v41_gfx90a/run_v4_serve.sh   # garbage
# phase 1: correct V4-Flash on two nodes (PP=2 x TP=8)
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 MHC_FIX=triton TRITON_LOGITS=1 NO_ASYNC=1 BF16_LINEAR=1 COMPILE=1 \
  MORE_VLLM_ARGS="--max-num-seqs 64" PROBE_GSM8K_N=500 BENCH=1 CONCURRENCIES="1 4 16 32 64" \
  sbatch --nodes=2 --time=03:00:00 repro/deepseek_v41_gfx90a/run_v4_serve.sh
# V4.1 on one node through the plugin: stage the checkpoint on LUMI-F once, then serve
sbatch --account=project_462000131 repro/deepseek_v41_gfx90a/stage_flash.sh
V41=1 AMDGCN_USE_BUFFER_OPS=0 BF16_LINEAR=1 COMPILE=1 TRITON_LOGITS=1 MHC_TRITON=1 CUSTOM_AR=1 NO_ASYNC=1 \
  MORE_VLLM_ARGS="--max-num-seqs 64" DIAG=1 PROBE_GSM8K_N=200 BENCH=1 CONCURRENCIES="1 32 64" \
  READY_TIMEOUT=5400 \
  sbatch --time=02:30:00 repro/deepseek_v41_gfx90a/run_v4_serve.sh
# one-GCD kernel tests for the two Triton replacements
sbatch repro/deepseek_v41_gfx90a/run_paged_logits_test.sh
sbatch repro/deepseek_v41_gfx90a/run_mhc_triton_test.sh
```
