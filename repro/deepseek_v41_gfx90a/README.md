# DeepSeek-V4.1-Flash on LUMI-G (MI250X, gfx90a): phase 0

Should we try to port `deepseek-ai/DeepSeek-V4.1-Flash` to LAIF's vLLM 0.26, as we did for
Qwen3.8-Flash (branch `exp/qwen38-flash-gfx90a`)? This is the de-risking pass, before any
port work and before downloading its 475 GiB.

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

## Reproduce

```bash
# from the repository root on LUMI
sbatch --partition=dev-g repro/deepseek_v41_gfx90a/run_kernel_tests.sh
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 sbatch repro/deepseek_v41_gfx90a/run_v4_serve.sh   # garbage
# phase 1: correct V4-Flash on two nodes (PP=2 x TP=8)
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 MHC_FIX=triton TRITON_LOGITS=1 NO_ASYNC=1 BF16_LINEAR=1 COMPILE=1 \
  MORE_VLLM_ARGS="--max-num-seqs 64" PROBE_GSM8K_N=500 BENCH=1 CONCURRENCIES="1 4 16 32 64" \
  sbatch --nodes=2 --time=03:00:00 repro/deepseek_v41_gfx90a/run_v4_serve.sh
```
