# DeepSeek-V4.1-Flash on LUMI-G (MI250X, gfx90a): phase 0

Should we try to port `deepseek-ai/DeepSeek-V4.1-Flash` to LAIF's vLLM 0.26, as we did for
Qwen3.8-Flash (branch `exp/qwen38-flash-gfx90a`)? This is the de-risking pass, before any
port work and before downloading its 475 GiB.

> **Status (2026-11-20): with DSpark speculative decoding (`DSPARK=5`, V4.1's own drafter)
> and kernels retuned for its 6-token verify batches: 64.5 tok/s at 1 user (job 23871589);
> 484.7 / 732.1 / 932.3 at 16 / 32 / 64 users in steady state (4 prompts per slot, job
> 23902280); 970 at 128 users with `--max-num-seqs 128`. Natural prompts (GSM8K questions),
> 1.7–2.0× the recipe without DSpark. GSM8K 98–99 %, needles. KV cache 2.24M tokens. Prefill (time to first token)
> 1.19 / 4.35 / 8.07 s at 4k / 16k / 30k after the Triton indexer logits,
> `SPARSE_PREFILL=1`, blocked FP8 rounding and prefill MoE tiles (job 23900916).**
>
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
| split-K sparse decode instead (`SPARSE_SPLIT=1`, no `CUSTOM_AR`) | 23698452 | **16.9** / – / 194.3 / 264.5, 402.2 at 64 | **59** / 108 ms, 146 at 64 |
| both (`CUSTOM_AR=1 SPARSE_SPLIT=1`) | 23700637 | 17.8 / 69.2 / 199.5 / 282.5, 422.2 at 64 | 55 / 100 ms, 138 at 64 |
| split-K up to 32 queries (`SPARSE_SPLIT=32`) | 23706326 | 17.9 / – / 200.2 / 299.4, 422.2 at 64 | 55 / 95 ms, 138 at 64 |
| + query RoPE in one Triton kernel (always on) | 23706895 | 18.7 / – / 207.1 / 308.9, 429.9 at 64 | 53 / 91 ms, 135 at 64 |
| same, async scheduling (no `NO_ASYNC=1`) | 23707780 | 20.9 / – / 225.3 / 332.4, 460.0 at 64 | 47 / 78 ms, 120 at 64 |
| **gfx90a sparse decode kernel** (`SPARSE_DECODE=1` instead of `SPARSE_SPLIT=32`), `NO_ASYNC=1` | 23708163 | **20.3 / – / 220.1 / 328.3**, **462.7 at 64** | **48 / 85 ms**, 124 at 64 |
| same, async scheduling (no `NO_ASYNC=1`) | 23783297 | 22.9 / – / 242.3 / 352.0, 494.2 at 64 | 43 / 73 ms, 111 at 64 |
| async + 2048-token chunks (`--max-num-batched-tokens 2048`) | 23851799 | 22.9 / – / 245.3 / 362.1, 515.2 at 64 | 43 / 83 ms, 117 at 64 |
| + `pack_bitmatrix` in 16-row blocks (`MOE_SMALL=1`) | 23851800 | 24.2 / – / 254.4 / 370.9, 526.2 at 64 | 41 / 81 ms, 114 at 64 |
| **+ V4.1 expert GEMM tiles** (`MOE_TILES=1`): the recipe | 23857167 | **35.8 / – / 263.4 / 395.0**, **542.5 at 64** | **27 / 75 ms**, 111 at 64 |
| + router GEMM via score_mm (`GATE_GEMM=1`), not adopted | 23857581 | 34.9 / – / 265.3 / 395.5, 546.3 at 64 | 28 / 75 ms, 110 at 64 |

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
- **Split-K sparse decode** (`SPARSE_SPLIT=1`, `hooks.py:_sparse_split`): V4's
  `sparse_split.py` on V4.1's vendored ops. Off gfx942/950 upstream decodes with one
  workgroup per query and head block; the hook takes the generic gfx942 split path (partial +
  reduce, CU-count heuristic, not the gfx950 kernels) for batches of at most 16 queries.
  `test_sparse_split.py` (job 23698377, V4.1's per-rank shapes = V4's): within 0.6 % of the
  fallback; ratio-4 layer 407 → 62 µs at 1 query, 424 → 149 at 16, but 447 → 611 at 64,
  hence the cutoff. Serving (job 23698452): **+20 % at 1 user (TPOT 71 → 59 ms)**, 32 and 64
  users unchanged (V4 lost 3 ms at 16–32; V4.1 does not). GSM8K 97.5 % of 200, but all five
  misses also occur in one of the other three runs (93, 100, 119, 161, 184): noise. Needles,
  decode = prefill as before.
- **Both together** (job 23700637): GSM8K **99.0 %** of 200 (misses 93 and 119, both seen
  before), both needles, decode = prefill to 19.7k, KV 1.35M tokens. 1.27× the previous
  recipe at 1 user, 1.07× at 32, 1.06× at 64. V4's recipe on the same node: 16.3 / 62.7 /
  192.4 / 319.5 / 466.8 tok/s at 1 / 4 / 16 / 32 / 64, so V4.1 is now ahead up to 16 users
  and 12 % / 10 % behind at 32 / 64.
- **Graph-mode profile at 32 users, V4.1 vs V4** (jobs 23701511 / 23706084, same recipe
  switches, `PROFILE=1` with graphs on; `analyze_trace.py` now classifies graph-replayed
  kernels by name). GPU ms per decode step, rank 0:

  | component | V4 | V4.1 |
  |---|---|---|
  | sparse attention (gfx90a fallback kernel) | 9.8 | **19.3** |
  | MoE experts (`matmul_ogs`, `pack_bitmatrix`) | 15.2 | 14.0 |
  | dense GEMMs (hipBLASLt) | 14.6 | 11.4 |
  | all-reduce (custom) | 7.2 | 4.8 |
  | elementwise / copies | 5.3 | **8.3** |
  | mHC | 4.9 | 4.0 |
  | compressor | 2.1 | ~0 |
  | FP8 rounding, q norm/RoPE | 2.0 | 2.3 |
  | indexer | 1.2 | 1.3 |
  | GPU busy per step (share of span) | 64.9 (81 %) | 67.6 (82 %) |

  V4.1 has 38 top-k-512 attention layers (compress ratios 1 and 2) to V4's 21 (ratio 4; its
  20 ratio-128 layers attend to a few dozen entries), so the one-pass fallback, one
  workgroup per query on a 110-CU GCD and ~50 GB/s effective, costs twice as much. The
  extra elementwise time is most likely `lumi_dsv41/ops.py`'s torch query RoPE (~10 small
  strided kernels per layer).
- **Split-K up to 32 queries** (`SPARSE_SPLIT=32`, job 23706326): at 32 queries the split
  path is 438 → 271 µs per top-k layer (job 23698377), × 38 layers. **+6.0 % at 32 users**
  (TPOT 100 → 95 ms), 1 / 16 / 64 unchanged, GSM8K 99.0 % (same two misses), needles,
  decode = prefill. V4 lost 10 ms at 32 users with the same cutoff (unexplained); V4.1,
  with almost twice the top-k layers, gains. Above 32 queries the split path is slower
  than the fallback (611 vs 447 µs at 64), so 64 users need a better kernel.
- **Query in Triton** (`lumi_dsv41/ops.py:_query_kernel`, always on): V4.1's query (head
  padding, optional norm, GPT-J RoPE on the last 64 dims) was ~10 strided torch kernels per
  layer around 0.26's fused KV kernel. `test_ops.py` (job 23706874): equal to the torch
  reference or within one bf16 step, 36.9 → 2.1 µs per call at 32 tokens. Serving (job
  23706895): **+4.5 % at 1 user, +3.4 % at 16, +3.2 % at 32, +1.8 % at 64**, more than the
  1.5 ms of kernel time (fewer launches between graph pieces), GSM8K 99.0 % (misses 119
  and 184, both seen before), needles, decode = prefill.
- **Async scheduling on this recipe** (job 23707780): GSM8K 98.5 % (misses 93, 119, 161, all
  seen before), needles, decode = prefill; **+11.7 % at 1 user, +8.8 % at 16, +7.6 % at 32,
  +7.0 % at 64**, a little more than on the older recipe (+7–9 %). KV cache 0.77M tokens
  instead of 1.35M, as before: the choice stays by workload (async for many short requests,
  `NO_ASYNC=1` when many long contexts must fit).
- **gfx90a sparse decode kernel** (`lumi_dsv41/sparse_decode.py`, `SPARSE_DECODE=1`,
  `hooks.py:_sparse_decode`). The stock fallback and upstream's split-K path share one
  inner loop: 16 entries per iteration, the UE8M0 scale byte gathered per element, NoPE
  and RoPE as separate dots with 448 padded to 512. The new partial kernel walks main and
  extra as one list split over a batch-dependent number of programs, loads 8 scale bytes per
  token, and builds one [16, 512] bf16 K tile (448 dequantised + 64 RoPE) per step: one
  QK^T and one PV dot over the 8 heads (padded to 16); a second kernel merges the splits
  with the sink. `test_sparse_decode.py` (jobs 23707915 sweep, 23707993 final): within
  bf16 rounding of the fallback (fixed and ragged lists, -1 slots, NaN in unused bytes,
  with and without sink), hook == direct call; GPU µs per top-k layer call:

  | queries | 1 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|
  | fallback | 410 | 421 | 438 | 447 | 891 |
  | upstream split-K | 62 | 151 | 281 | 584 | 1266 |
  | **this kernel** | **30** | **56** | **108** | **189** | **339** |

  The FP8 decode is ~20 % of it (86 vs 108 µs at 32 queries without it). Serving (job
  23708163): **+8.7 % at 1 user, +6.3 % at 16 and 32, +7.6 % at 64** over the split-K
  recipe; GSM8K 98.5 % (misses 93, 119, 161, all seen before), needles, decode = prefill.
  V4.1 now matches or passes V4's recipe at every load (V4 with `SCORE_GEMM=1`: 17.7 /
  198.1 / 325.1 / 481.8 at 1 / 16 / 32 / 64).
- **Kernel + async scheduling** (job 23783297): **22.9 / 242.3 / 352.0 / 494.2 tok/s** at
  1 / 16 / 32 / 64 users (+12.7 / +10.1 / +7.2 / +6.8 % over the synchronous kernel run),
  needles and decode = prefill fine, KV 0.77M tokens. GSM8K 97.5 % of 200: misses 100, 119,
  161, 184 (seen before) and 125, which none of the 15 earlier good V4.1 runs missed, nor the
  synchronous kernel run nor async without the kernel. The 500-question check (job
  23784396) gives **GSM8K 98.2 %** (9 misses; 125 right this time, 93 and 119 as always),
  above V4's recipe on the same 500 (97.0–97.8 %): batching nondeterminism, not a defect.
  Validated. (A first attempt, job 23783775, died at start-up with "HW Exception by GPU
  node-4: GPU Hang" during the memory-profiling forward on nid005019; the same code started
  cleanly on three other nodes.)
- **Async with 2048-token chunks** (from V4, "Async scheduling, revisited"; job 23851799):
  vLLM reserves window + `max_in_flight_tokens` per request in every sliding-window layer,
  and `max_in_flight_tokens` = batches in flight × `max_num_batched_tokens`. Async with
  8192-token chunks reserved 2 × 8192, hence 0.77M tokens of KV; with 2048 it reserves
  2 × 2048 and the KV cache grows to **2.43M tokens** (74 concurrent 32k requests; sync with
  8192 had 1.35M). Throughput 22.9 / 245.3 / 362.1 / 515.2 tok/s at 1 / 16 / 32 / 64 users
  (+0 / +1 / +3 / +4 % over async with 8192), mean TTFT 2.8 / 4.4 / 7.5 s at 16 / 32 / 64
  (4.4 / 7.5 / 11.4 s before). GSM8K 98.0 % (misses 93, 100, 119, 184, all seen before),
  needles at 2.1k and 16.8k, decode = prefill to 19.7k. **Async is now better on every
  axis, so the recipe drops `NO_ASYNC=1`; no workload choice left.**

  The first try crashed (jobs 23836592/3): `_forward_prefill` got a chunk of 22 prefills
  with zero lengths and failed on `view(22, 0, -1)`. The guard `_empty_prefill_chunks`
  (hooks.py) skips such chunks and logged the batch: the last 78-token chunk of the
  2,126-token needle prompt, alone, padded by 0.26's runner to 64 requests (80 tokens);
  `num_decodes=0 num_prefills=64 query_lens=[78, 0, …]`. Upstream counts the 63 padding
  requests as prefills; with 8192-token chunks the prompt never reached a graph-sized
  batch alone. Padding requests own no tokens, so skipping their chunk is exact (the needle
  run itself is the check). The third 0.26 padding mismatch after the indexer one.
- **`MOE_SMALL=1`, routing pack only** (job 23851800; V4's `moe_small.install_pack`, via
  `DSV41_MOE_PACK` in `lumi_dsv41/__init__.py`): vLLM's `make_routing_data` launches
  `pack_bitmatrix` with 512-row blocks whatever the batch; 16-row blocks give the same
  bitmatrix. **+5.4 % at 1 user, +3.7 % at 16, +2.4 % at 32, +2.1 % at 64.** GSM8K 98.0 %
  (misses 100, 119, 184 as before, 125 as in job 23783297), needles, decode = prefill.
- **V4.1 expert GEMM tiles** (`MOE_TILES=1`, `lumi_dsv41/__init__.py`; V4's `moe_small`
  `matmul_ogs` wrapper with V4.1's tables). V4's tiles are for its TP-sharded experts (N 512
  per rank); V4.1 runs 48 whole experts per rank under EP. `test_moe_matmul.py` replays
  the EP call (global top-6 of 384 mapped to the rank's experts, -1 elsewhere, buffers of
  tokens × 6 rows of which only the routed ones are written), cycling 8 routings so the
  1.1 GB of local expert weights stay cold (job 23857141; the first run, 23857033, compared
  the unwritten rows too and showed NaN). Default flags vs forced block_n / split_k, µs:

  | tokens | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|---|---|
  | gate/up 5120 → 4608 | 143 → **38** (32/2) | 201 → **60** (32/4) | 154 → 72 (128/2) | 233 → 168 (128/2) | 301 → 237 (128/1) | 513 → 431 | 807 → 666 | 1006 → 827 |
  | down 2304 → 5120 | 112 → **32** (32/2) | 113 → **41** (32/2) | 66 → 48 (128/2) | 98 → 82 (128/1) | 130 → 121 | 208 → 198 | 336 → 327 | 446 → 441 |

  Same output as the default tiles on the routed rows (≤ 2e-3 with split-K, exact without).
  Serving (job 23857167): **35.8 tok/s at 1 user (TPOT 41 → 27 ms, +48 %)**, +3.5 % at 16,
  +6.5 % at 32, +3.1 % at 64; GSM8K 98.0 % (misses 93, 100, 119, 184, all seen before),
  needles, decode = prefill. At one token the default launch is sized for every expert's
  grid and takes ~140 µs whether a rank gets 0, 1 or 2 routed rows; the ranks then wait for
  the slowest at the all-reduce, so the gain under graphs exceeds the per-call sum. The
  warm-up phase (4 users) includes the new tiles' first compilation, so its TPOT is not a
  measurement.
- **Router GEMM** (`GATE_GEMM=1`: V4's `GateLinear` → `score_gemm.score_mm` with V4.1 tiles
  for 5120 → 384, `lumi_dsv41/__init__.py`). `test_router_gemm.py` (job 23857491): today's
  bf16 `F.linear` + cast is 37 µs a call with the logits rounded to bf16 (rel err 2–3e-3);
  `score_mm` is 13–21 µs at 1–128 tokens with fp32 logits (2e-7). In serving (job
  23857581) the ~1 ms a step it saves does not show: 34.9 / 265.3 / 395.5 / 546.3 tok/s
  (−2.7 / +0.7 / +0.1 / +0.7 % against job 23857167, within run-to-run spread), and GSM8K
  97.0 % with one question never missed before (16) besides 93, 100, 119, 161, 184. Off:
  no measurable gain, and it changes the routing.

### Speculative decoding: scoping (2026-11-19)

V4.1's `mtp.{0,1,2}` weights are a **DSpark** drafter (`vendor/.../deepseek_v41/amd/dspark.py`),
not V4-style MTP: one parallel pass drafts a block of 5 tokens (`dspark_block_size`) with
non-causal attention, then a Markov head (`dspark_markov_rank` 256) samples them left to
right. It reads the target's hidden states after layers 37–39 (`dspark_target_layer_ids`).
Its 3 layers are SWA-only (compress ratio 0: no indexer, no compressor) with their own MoE
(128 experts, top-3). What vLLM 0.26 already has, and what is missing:

| piece | 0.26 | needed |
|---|---|---|
| DSpark speculator in MRV2 (`v1/worker/gpu/spec_decode/dspark`, on DFlash's context-KV + query-block machinery, FULL graphs for the draft step) | yes | — |
| `--speculative-config '{"method": "dspark"}'` with the drafter in the target checkpoint | yes: sets the draft arch to `DSparkDraftModel` | register `DSparkDraftModel` → the vendored V4.1 class |
| hooks the speculator calls (`compute_draft_logits`, `markov_embed`, `markov_bias`, `map_draft_to_target`, `combine_hidden_states`, `precompute_and_store_context_kv`, `get_draft_kv_cache_layer_names`) | — | all in the vendored drafter |
| `qwen3_dspark.DSparkMarkovHead` | yes, same signature | — |
| `qwen3_dspark.DSparkConfidenceHead` | **no** | graft (upstream's adaptive verification; 0.26's speculator never calls it) |
| target aux hidden states | — | vendored V4.1 model has `aux_hidden_state_layers` |
| target verification: 6 query tokens per sequence | — | the V4.1 indexer flattens >2 tokens to 1 per row on ROCm (`_use_flattening`), so our Triton logits kernel applies; sparse decode kernel takes any row count |
| non-causal SWA indices for the draft block | — | vendored (`ComputeDSparkNoncausalSWAIndicesKernel`); untested on gfx90a |
| plugin refuses any speculative config (`verify_and_update_config`) | — | lift for dspark |

Plan: (1) eager smoke with DSpark (registrations, graft, the refusal lifted), check
acceptance and GSM8K; (2) graphs: the draft step is FULL-captured, so every op in the
draft layers must be capturable (our kernels are; any `.item()` fallback is not); (3)
0.26 padding mismatches under spec decode (the indexer and prefill-chunk hooks assume one
query per sequence in places); (4) block size (5, or fewer at high load) and async.
V4's MTP gave 1.8× at 1 user with ~75 % acceptance of one draft; DSpark drafts 5 per step.

**Step 1 done: DSpark runs, eager** (`DSPARK=5`, job 23862125; `lumi_dsv41/dspark.py`).
Five 0.26-vs-upstream gaps on the way, one per run:

| job | failure | fix |
|---|---|---|
| 23861137 | `replace(attention_config)`: field `sparse_mla_prefill_workspace_factor` not found | `hooks.py:_config_replace` rebuilds from fields, carries the plugin's extra attributes |
| 23861245 | `indexer_kv_dtype='auto'` fails 0.26's Literal on rebuild | carried the same way |
| 23861528 | `KeyError: model.layers.0.ffn.gate.bias_vl` | drafter skips the vision routing bias, as the target |
| 23861746 | 0.26's runner slices `get_mtp_target_hidden_states()`, which upstream returns as None for DSpark | target returns the step's hidden states then |
| 23861880 | context-KV insert calls the fused op with upstream's 11 arguments (0.26: 9) | `_v41_dspark` routes it through `ops.py` |

Result: sanity 5/5, needles at 2.1k and 16.8k, **GSM8K 99 % of 100**, KV 2.24M tokens.
Acceptance (vLLM's SpecDecoding metrics, 5 drafts): during GSM8K (natural text) **mean
acceptance length 3.7–4.0 of 6**, first draft accepted 82–89 %, per position ~0.87 / 0.71
/ 0.55 / 0.42 / 0.30; during the benchmark, whose prompts are random words, 1.2–2.4. So
the random-prompt sweep understates speculative decoding badly; step 2 needs a
natural-text benchmark. Eager speed is not a measurement (8.4 tok/s at 1 user on the
random prompts).

**Step 2 done: DSpark with graphs** (`COMPILE=1 DSPARK=5`, job 23863557; baseline job
23864381, same recipe without DSpark). The DSpark draft step is FULL-captured by 0.26's
DFlash graph manager and runs on gfx90a without further changes. Both runs use
`BENCH_DATA=gsm8k` (new: 512 GSM8K test questions after the probe's, "Solve it step by
step.", up to 512 output tokens, stopping at EOS), since random-token prompts understate
speculative decoding:

| users | 1 | 16 | 32 | 64 |
|---|---|---|---|---|
| recipe (tok/s) | 37.2 | 220.5 | 349.5 | 514.6 |
| **+ DSpark, 5 drafts (tok/s)** | **66.0** | **376.4** | **553.5** | **755.6** |
| speedup | **1.77×** | **1.71×** | **1.58×** | **1.47×** |
| TPOT | 26 → 14 ms | 52 → 26 ms | 66 → 39 ms | 91 → 66 ms |
| mean TTFT | 169 → 141 ms | 644 → 461 ms | 656 → 795 ms | 902 → 1251 ms |

GSM8K 99.0 % with DSpark (98.0 % without), sanity 5/5, needles at 2.1k and 16.8k. Mean
acceptance length 3.2–3.9 of 6 over the sweep (45–57 % of drafts accepted). KV cache
2.24M tokens with the drafter's layers, 2.43M without.

**Tuning for DSpark's verify batches** (6 tokens per sequence, so 96–384 rows at 16–64
users). Graph profile at 32 users with DSpark (job 23871174, `BENCH_DATA=gsm8k` prompts;
`analyze_trace.py` now marks spec-decode launches `[draft]`): GPU busy 95 %, the draft step
~9 ms of ~118 ms; MoE experts 35.5 ms (`matmul_ogs` on the library's default tile, since
192 tokens was past `MOE_TILES`'s table), dense GEMMs 18.7, all-reduce 14.3 (custom kernel,
172 µs per 2 MB call), elementwise 13.0, sparse attention 10.3; and the mHC prenorm GEMM in
torch above 64 tokens (fp32 copy + fp32 GEMM + square-sum + split-K reduce, ~99 µs per
call, 80 calls). Sweeps: gate/up with block_n 128 / split_k 1 is 13–16 % faster from 96 to
768 tokens, exact (job 23871479: 1068 → 933 µs at 192, 1722 → 1458 at 384); mHC split-K
beats the torch GEMM to ~256 tokens (job 23871480: 0.195 → 0.099 ms at 96, 0.197 → 0.155
at 192, equal at 384). With `MOE_TILES` to 1024 tokens and mHC split-K to 256 (job
23871589): **64.5 / 446.3 / 621.5 / 754.4 tok/s** at 1 / 16 / 32 / 64 users (−2 / **+19** /
**+12** / 0 %), GSM8K 98.0 %, sanity 5/5, needles. 64 users did not move, although the
gate/up sweep promised ~10 ms a step there; not investigated.

**64 users, steady state, and a tiled mHC prenorm GEMM.** Graph profile at 64 users with
DSpark (job 23880494): the extended MoE tile is in use (`matmul_ogs ...64x128x128`, 657 µs a
call); a step is ~195 ms and yields ~211 tokens (~1,080 tok/s at full occupancy), while the
sweep with 2 prompts per slot measured 754: requests ending at different lengths leave a long
tail with fewer than 64 active. With `PROMPTS_PER_SLOT=4` (job 23882337): 560.5 / 830.2
tok/s at 32 / 64 users. At 384 rows the mHC prenorm GEMM was back on the torch fallback
(~12 ms a step): the split-K kernel is a vector sum per token. `_prenorm_gemm_tiled_kernel`
(`mhc_triton.py`, from 96 tokens) does a token block per program with an fp32 matrix-core
dot (IEEE, fn not rounded): exact against the reference (≤ 5e-7, layer-0 form included),
whole delayed-pre call 0.154 → 0.100 ms at 192 tokens, 0.293 → 0.163 at 384, 0.548 → 0.293
at 768 (job 23882347). Serving, 4 prompts per slot (job 23895024): **501.3 / 684.2 / 876.7
tok/s at 16 / 32 / 64 users**, GSM8K 99.0 %, sanity 5/5, needles. +5.6 % at 64; the +22 % at
32 overstates it (the baseline's 32-user phase ran first after warm-up, mean TTFT 2.8 s vs
0.6 s here).

**All-reduce, sparse-decode splits and NUMA binding at verify sizes** (4 prompts per slot,
16 / 32 / 64 users). `test_allreduce.py` on 8 GCDs at hidden 5120 (job 23896111): the custom
kernel (copy mode, exact) beats RCCL at every DSpark size, 26.9 vs 95.9 µs at 32 tokens,
142.7 vs 193.5 at 384, 264.9 vs 302.2 at 768; no switch. In serving a call takes ~279 µs at
384 tokens, so about half of it is ranks waiting for the slowest. Sparse decode beyond 128
queries (job 23896112): 4 splits at 192 (473 vs 504 µs), 2 at 384 (874 vs 985); the rule now
aims at ~1024 workgroups there (`sparse_decode.num_splits`). Serving (job 23897202): 511.6 /
711.0 / 855.3 tok/s (+2 / +4 / −2 % against 501.3 / 684.2 / 876.7, job 23895024; within run
spread), GSM8K 98.5 %. `NUMA_BIND=1` on top (job 23897203; workers bound to their GCD's CCD
and NUMA node, confirmed in the log): 479.0 / 697.6 / 882.4 (−6 / −2 / +3 %), GSM8K 97.5 %:
no gain; left off.

**Prefill (time to first token).** Agentic turns resend 10–30k-token contexts, and nothing
so far had touched prefill. One user, 16 output tokens (job 23898195): TTFT 1.66 / 6.83 /
13.93 s at 4k / 16k / 30k prompt tokens (~2,400 tokens/s). Prefill profile of one 16.5k
prompt (job 23898197, `PROFILE_PREFILL=16384`, new `profile_prefill.py`; ~547 ms per
2048-token chunk, GPU busy 99 %): sparse-attention prefill 156 ms (upstream's ROCm kernel,
4.65 ms a call), the indexer's prefill logits in torch ~85 ms (`fp8_mqa_logits_torch`:
FP8 casts, fp32-output GEMM, elementwise, reduction), RCCL all-reduce 70 ms (20 MB
messages), MoE 75 ms, mHC ~50 ms, FP8 rounding 30 ms.

- **Indexer prefill logits in Triton** (`lumi_dsv41/mqa_logits.py`, `PREFILL_LOGITS=1`): a
  [32 rows × 128 keys] tile per program, the key block loaded once and reused across the 32
  heads (bf16 MFMA, rounded to bf16 like the einsum), tiles outside every row's causal
  range written as -inf without computing. `test_mqa_logits.py` (job 23900362), 2048 rows:
  4.52 / 18.17 / 37.80 / 68.11 ms → 0.33 / 1.90 / 4.03 / 7.50 ms at 2k / 8k / 16k / 30k keys;
  same mask, max rel err 2e-7, top-512 overlap ≥ 99.96 %.
- **Sparse-attention prefill without software pipelining** (`SPARSE_PREFILL=1`,
  `hooks.py:_prefill_stages`): upstream's wrapper leaves Triton's default `num_stages=2`;
  with 1, same tile, the same output: 5.98 → 2.73 ms per 2048-query chunk at 4k, 5.99 →
  2.87 at 16k (`test_sparse_prefill.py`, job 23900392; every pipelined setting ~5.8 ms).
- All-reduce at prefill sizes (job 23900284): RCCL 236 / 381 µs vs the custom kernel
  347 / 657 µs at 1024 / 2048 tokens, so vLLM's 8 MB limit already routes them right
  (`CUSTOM_AR_MAX_MB` added, left unset).

| prompt tokens | 4k | 16k | 30k |
|---|---|---|---|
| before (job 23898195) | 1.66 s | 6.83 s | 13.93 s |
| `PREFILL_LOGITS=1` (job 23900387) | 1.51 s | 5.90 s | 10.92 s |
| **+ `SPARSE_PREFILL=1`** (job 23900437) | **1.32 s** | **4.86 s** | **9.16 s** |

| **+ blocked FP8 rounding, MoE tiles to 8192 tokens** (job 23900916) | **1.19 s** | **4.35 s** | **8.07 s** |

−21 / −29 / −34 % after the first two, −28 / −36 / −42 % in all; GSM8K 98.5 / 99.0 %, sanity
5/5, needles at 2.1k and 16.8k. The last row: the FP8 activation rounding launched one
1-warp program per 32-column group (V4.1's 32×32 blocks), mostly launch overhead; a blocked
kernel (2048 columns a program, per-group maxima, the same arithmetic; V4 plugin
`bf16_linear.py`) is bit-identical over every case of `test_fp8_round.py` and 438 → 45 µs at
2048 rows, 84 → 11 at 384, 8.9 → 2.8 at 32 (job 23900869). MoE gate/up with block_n 128 /
split_k 1 is also 13–14 % faster at 1024–2048 tokens (job 23900870), so `MOE_TILES` covers
prefill chunks too. mHC at prefill sizes is already on the tiled path (0.71 ms a call at 2048
tokens, twice the alternatives; job 23900871). Decode with the same changes (job 23902280, natural prompts, 4 per
slot): 64.9 / 484.7 / 732.1 / 932.3 tok/s at 1 / 16 / 32 / 64 users (+1 / −5 / +3 / **+9 %**;
the 64-user gain is the rounding at 384 rows, 84 → 11 µs a call; 16 users has ranged
484–512 across runs of unchanged code).

Decode at 1 user with DSpark (profile job 23898196) is host/rank-skew bound rather than
GPU bound: all-reduces wait ~2 ms each in the draft step (12 a step) and ~0.2 ms in the
verify step (81), against ~27 µs of kernel; the profiler's own host overhead (37 tokens in
3 s) exaggerates it.

**Prefix caching: scoping (2026-11-21).** An agent resends its growing context each turn, so
every turn re-prefills it (8 s at 30k tokens). The block is ours, not 0.26's:
`lumi_dsv41/__init__.py` refuses `--enable-prefix-caching`, and the compressor ring's
manager (`circular_buffer.py`) always reports a 0-token hit, which the hybrid coordinator
takes as the minimum over all groups. Upstream V4.1 rebuilds sliding-window and compressor
state on a hit by replaying the window before it (`swa_bounded_replay`, decoder replay):
engine-side work 0.26 lacks, but upstream itself turns replay off on ROCm below CDNA3
(`vendor/.../deepseek_v41/attention.py:605-614`, "the sliding-window cache takes part in
prefix caching instead"), which is the mode 0.26 implements. Per-request state at a
block-aligned hit: compressed KV and indexer K (`MLAAttentionSpec`, 0.26's
`FullAttentionManager`) reusable; SWA KV (`SlidingWindowMLASpec`) reusable from 0.26's
retained tails; the ratio-2 compressor ring is only read at odd positions
(`fused_compress_quant_cache.py:174`), so an even, block-aligned hit needs a fresh ring;
Engram lookback is rebuilt from the token history each step; mHC has no cross-token state.
Plan: a neutral ring manager (hit = the candidate length, rounded to its alignment), the
refusal behind an opt-in switch, an assert that hits are even, and a test of hit vs no-hit
outputs (chunked prefill, preemption, DSpark). Open risk: whether 0.26's DSpark speculator
writes the drafter's context KV for every prefilled prompt token.

**128 sequences** (`--max-num-seqs 128`, with DSpark, job 23870587): **970 tok/s at 128
users** (TPOT 98 ms, mean TTFT 2.3 s), 548 / 729 at 32 / 64 (−1 / −4 % against 64
sequences: more graph sizes and 2.05M instead of 2.24M tokens of KV cache). GSM8K 98.5 %,
sanity 5/5, needles; acceptance length 3.1–3.4 at 128 users. So `--max-num-seqs 128` for
throughput-bound serving, 64 otherwise.

**Bit-level FP8 decode in the sparse kernel** (`sparse_decode.py`, `deq=2`, off): e4m3 ×
UE8M0 written straight into bf16 bits instead of Triton's software FP8 conversion and an
fp32 multiply. Bit-identical to the default path over every e4m3 code (jobs 23870597 /
23870620), but slower from 16 queries (top-k layer at 32: 86 → 108 µs, at 128: 210 → 267),
faster only at 1–16 on SWA layers. Not used. (The test's stock-kernel comparison excludes
the NaN codes 0x7f / 0xff, which the stock kernel decodes as finite values and the cache
writer never produces.)

**Draft count:** 5 is the only working setting. `DSPARK=3` is refused by 0.26 itself
(job 23866836: "DSpark requires num_speculative_tokens >= dspark_block_size (5) ... Smaller
values produce incorrect output"), so 4 is too. `DSPARK=7` (job 23868852) fails at start-up
in 0.26's DeepSeek-V4 KV page packer (`_get_kv_cache_groups_uniform_groups`, an assertion
on the group layout with the larger draft reservation); not pursued, since the drafter is
trained on 5-token blocks and extra drafts mostly add verification work at load.

Open: the FP8 decode in the sparse kernel (~20 %, gfx90a converts in software); FULL decode graphs (no gain on V4); reporting the two upstream gfx90a
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
V41=1 AMDGCN_USE_BUFFER_OPS=0 BF16_LINEAR=1 COMPILE=1 TRITON_LOGITS=1 MHC_TRITON=1 CUSTOM_AR=1 SPARSE_DECODE=1 MOE_SMALL=1 MOE_TILES=1 DSPARK=5 PREFILL_LOGITS=1 SPARSE_PREFILL=1 \
  MORE_VLLM_ARGS="--max-num-seqs 64 --max-num-batched-tokens 2048" PROBE_GSM8K_N=200 BENCH=1 BENCH_DATA=gsm8k CONCURRENCIES="1 16 32 64" \
  READY_TIMEOUT=5400 \
  sbatch --time=02:30:00 repro/deepseek_v41_gfx90a/run_v4_serve.sh
# one-GCD kernel tests for the two Triton replacements
sbatch repro/deepseek_v41_gfx90a/run_paged_logits_test.sh
sbatch repro/deepseek_v41_gfx90a/run_mhc_triton_test.sh
sbatch repro/deepseek_v41_gfx90a/run_sparse_split_test.sh
sbatch repro/deepseek_v41_gfx90a/run_sparse_decode_test.sh
sbatch repro/deepseek_v41_gfx90a/run_ops_test.sh
sbatch repro/deepseek_v41_gfx90a/run_moe_matmul_test.sh
```
