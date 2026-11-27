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

For speed, add `COMPILE=1 TRITON_LOGITS=1 NO_ASYNC=1 BF16_LINEAR=1` (piecewise CUDA graphs,
the batched indexer kernel, bf16 dense linears with fused FP8 activation rounding), and
replace `TORCH_MHC=1` with `MHC_FIX=triton`, vLLM's fused TileLang hyper-connections with
my Triton kernel in place of the broken TileLang pre stage. `ROUTER_GUARD=1` is the default and must stay on: under CUDA graphs it is
what keeps NaN in pad rows out of the MoE router (see "A layout-sensitive bug under CUDA
graphs" below). `CUSTOM_AR=1` replaces RCCL with vLLM's custom all-reduce, made coherent on
gfx90a, for 1.03–1.11× (see "All-reduce" below). `SCORE_GEMM=1` runs the compressors'
fp32-output score GEMMs in a Triton kernel, 1.01–1.08× more (see "Compressor score GEMMs").
Async scheduling (no `NO_ASYNC=1`) with `--max-num-batched-tokens 2048` adds 1.05–1.08×,
3.4× the KV capacity and half the TTFT (see "Async scheduling, revisited"). `MTP=1` adds
speculative decoding with V4's MTP layer: **34 tok/s for one user** (from 18.7), 1.27× at
32 users and 1.09× at 64; `SPARSE_SPLIT=1 SPARSE_SPLIT_MAX=8` adds split-K sparse decode
for 1–4 users (see "MTP speculative decoding"). `MOE_SMALL=1 GATE_GEMM=1` retunes the MoE
routing and expert GEMMs for decode-sized batches and runs the router in Triton: 1.03–1.12×
at 4–32 users, ~1.02× at 1 and 64 (see "MoE at decode sizes"). `CUDAGRAPH_MODE=FULL_AND_PIECEWISE`
captures whole decode steps, which removes the host launch overhead that held back one rank
per step: **42–43 tok/s for one user** (from 37), unchanged from 4 users up (see "Where the
1-user step goes"). `SPARSE_DECODE=1` replaces `SPARSE_SPLIT` with the V4.1 port's sparse
decode kernel: **48 tok/s for one user**, 1.05–1.15× at every load (see "Sparse decode
kernel from V4.1"). `WO_A=1 DENSE_GEMM=1` run the output projection `wo_a` and the other
bf16 dense linears in Triton where it beats vLLM's ROCm GEMM dispatch: TPOT 1.3–3.6 ms
lower from 4 users up (see "Dense bf16 GEMMs at decode sizes"). `MHC_FAST=1` runs the
hyper-connections' post and prenorm GEMM in Triton (from the V4.1 port), closer to the
torch reference than TileLang and TPOT 0.4–1.2 ms lower from 4 users up (see
"Hyper-connections from the V4.1 port"). Use `--max-num-seqs 64` for every load: with
short natural prompts, 32 sequences was 13 % slower at 32 users (same section).
`MOE_SMALL`'s expert GEMM table v2 (default) and `MOE_GLUE=1` cut the MoE's small kernels:
**71 tok/s for one user** on GSM8K prompts, TPOT 0.9–1.3 ms lower at every load (see
"Small kernels around the MoE"). Its routing prologue now also covers the legacy
`triton_kernels` path this container takes, and `QNORM_ROPE=1` runs attention's q-norm /
RoPE / KV insert in one Triton kernel: **75.6 tok/s for one user**, 824 at 32 (see
"The routing prologue on the legacy path; q-norm / RoPE / KV insert"). For prefill,
`SPARSE_PREFILL=1` (V4.1's launch options for the sparse-attention prefill kernel) and the
score-GEMM tiles for prefill chunks (on by default) and `PREFILL_LOGITS=1` (the indexer's
prefill logits in Triton) and `MHC_FAST` at prefill sizes (on by default) take a 16k-token
prompt's time to the first token from 6.45 to **3.06 s** (see "Prefill"). `PREFIX_CACHING=1`
reuses resent prefixes (time to first token 3.2 s → 0.13–0.32 s on 17k-token prompts) and
gives an agentic load 7× the throughput (see "Prefix caching").

```bash
AMDGCN_USE_BUFFER_OPS=0 INDEXER_GATE=1 MHC_FIX=triton CUSTOM_AR=1 SCORE_GEMM=1 \
  MTP=1 SPARSE_DECODE=1 MOE_SMALL=1 GATE_GEMM=1 WO_A=1 DENSE_GEMM=1 MHC_FAST=1 MOE_GLUE=1 QNORM_ROPE=1 SPARSE_PREFILL=1 PREFILL_LOGITS=1 \
  PREFIX_CACHING=1 \
  TRITON_LOGITS=1 BF16_LINEAR=1 COMPILE=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  MORE_VLLM_ARGS="--max-num-seqs 64 --max-num-batched-tokens 2048" \
  sbatch repro/deepseek_v41_gfx90a/run_v4_serve.sh
```

Validated at both batch limits (500 GSM8K questions each, sanity 5/5, both needles, no
garbage, no runaway answers):

| hyper-connections | `--max-num-seqs` | job | GSM8K | out tok/s at 1 / 4 / 16 / 32 / 64 users | TPOT at 1 / 32 / 64 |
|---|---|---|---|---|---|
| **the command above**, GSM8K-question prompts (`BENCH_DATA=gsm8k`) | 64 | 23989320 | 97.6 % | **75.6 / 223.4 / 603.2 / 824.1 / 1144.3** | 13 / 31 / 47 ms |
| without the legacy-path routing prologue and `QNORM_ROPE` | 64 | 23983908 | 97.4 % | 70.7 / 210.3 / 576.2 / 783.5 / 1098.2 | 14 / 32 / 50 ms |
| without `MOE_GLUE` and MoE table v2, GSM8K-question prompts | 64 | 23898747 | 97.4 % | 57.0 / 184.7 / 513.4 / 709.5 / 1054.5 | 17 / 34 / 52 ms |
| the same, GSM8K-question prompts | 32 | 23898749 | 97.0 % | 54.3 / 181.5 / 485.8 / 618.5 / – | 17 / 42 / – ms |
| without `MHC_FAST`, 1024-token random prompts | 32 | 23896453 | 97.2 % | **51.8** / 141.3 / 360.9 / **538.7** / – | 18 / 49 / – ms |
| without `MHC_FAST`, 1024-token random prompts | 64 | 23880589 | 97.0 % | 44.8 / **169.7** / **364.4** / 524.1 / **717.1** | 22 / 50 / 73 ms |
| without `WO_A`, `DENSE_GEMM` | 32 | 23858696 | 97.2 % | 48.2 / 135.9 / 365.8 / 512.5 / – | 20 / 52 / – ms |
| without `WO_A`, `DENSE_GEMM` | 64 | 23857052 | 97.4 % | 48.2 / 141.1 / 346.9 / 494.2 / 707.0 | 20 / 52 / 75 ms |
| split-K sparse decode, full graphs | 32 | 23851784 | 97.4 % | 42.7 / 129.8 / 331.6 / 496.1 / – | 23 / 55 / – ms |
| split-K sparse decode, full graphs | 64 | 23836295 | 97.2 % | 42.0 / 120.3 / 308.6 / 499.3 / 648.1 | 23 / 54 / 81 ms |
| piecewise graphs, + MoE | 32 | 23811671 | 97.4 % | 35.2 / 122.1 / 333.6 / 501.9 / – | 28 / 54 / – ms |
| piecewise graphs, + MoE | 64 | 23811670 | 97.6 % | 35.4 / 128.0 / 306.2 / 464.9 / 643.9 | 28 / 54 / 80 ms |
| + async (2048-token chunks), MTP, split-K | 32 | 23802055 | 97.6 % | **34.3** / **118.9** / 307.2 / **446.2** / – | 29 / 57 / – ms |
| + async (2048-token chunks), MTP, split-K | 64 | 23802056 | 97.2 % | **35.1** / – / 296.2 / 411.0 / **571.2** | 28 / 57 / 83 ms |
| the same, run verbatim | 64 | 23810538 | 97.4 % | 34.8 / 114.0 / 275.2 / 431.2 / 630.3 | 28 / 59 / 85 ms |
| + async, 4096-token chunks | 32 | 23707179 | 97.8 % | 18.6 / 68.2 / 211.0 / 348.9 / – | 53 / 83 / – ms |
| + async, 4096-token chunks | 64 | 23707180 | 97.4 % | 18.6 / – / 211.3 / 349.1 / 521.7 | 53 / 84 / 114 ms |
| `MHC_FIX=triton` + `CUSTOM_AR=1` + `SCORE_GEMM=1` | 32 | 23706308 | 97.0 % | 17.7 / 64.7 / 198.3 / 323.3 / – | 56 / 89 / – ms |
| `MHC_FIX=triton` + `CUSTOM_AR=1` + `SCORE_GEMM=1` | 64 | 23706309 | 97.2 % | 17.7 / – / 198.1 / 325.1 / 481.8 | 56 / 88 / 122 ms |
| `MHC_FIX=triton` + `CUSTOM_AR=1` | 32 | 23696592 | 97.6 % | 16.3 / 62.7 / 192.4 / 319.5 / – | 61 / 90 / – ms |
| `MHC_FIX=triton` + `CUSTOM_AR=1` | 64 | 23696593 | 97.6 % | 16.3 / – / 192.5 / 318.2 / 466.8 | 61 / 90 / 126 ms |
| `MHC_FIX=triton` | 32 (default) | 23591543 | 97.4 % | 15.5 / 56.1 / 187.0 / 296.1 / – | 64 / 97 / – ms |
| `MHC_FIX=triton` | 64 | 23591544 | 97.8 % | 15.5 / 56.2 / 187.9 / **298.0** / **436.6** | 64 / 97 / 136 ms |
| `MHC_FIX=direct` | 32 | 23572794 | 97.4 % | 14.3 / 51.1 / 168.5 / 263.2 / – | 69 / 107 / – ms |
| `MHC_FIX=direct` | 64 | 23572795 | 97.8 % | 14.4 / 51.1 / 169.3 / 263.8 / 379.2 | 69 / 107 / 154 ms |
| `TORCH_MHC=1` | 32 | 23564811 | 97.6 % | 9.9 / 35.8 / 121.0 / 196.1 / – | 101 / 149 / – ms |
| `TORCH_MHC=1` | 64 | 23572796 | 98.0 % | 9.8 / – / – / 195.0 / 302.9 | 101 / 149 / 196 ms |

With MTP, 1- and 4-user rows vary by up to ±10 % between runs with the same speed per step:
the random-token prompts make acceptance shift with any numerical change (see "Dense bf16
GEMMs at decode sizes").

`MHC_FIX=triton` with 64 sequences is 5.0–7.3× eager. It is 1.53–1.58× the torch
hyper-connections at every load, and 1.08–1.15× `MHC_FIX=direct`, at the same GSM8K. 64
sequences gives 1.47× the 32-user throughput at ~6.8 tok/s per user. Each run took 500 GSM8K
questions plus the speed sweep: sanity 5/5, both needles, no garbage, no runaway answers.
`MHC_FIX=direct` drops TileLang from the hyper-connections entirely and is the fallback,
then `TORCH_MHC=1`. Run 23564811 predates the guard; it passed only because its pad rows
happened to stay finite.

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

### Dense FP8 linears: stock vs tuned vs bf16 (job 23535213)

`bench_fp8_linear.py`, one GCD, the five shapes vLLM warned about, CUDA-graph timings;
error is relative to fp32 activations times dequantised weights:

| N×K | M | stock µs | tuned µs | **bf16 µs** | error W8A8 | **error bf16** |
|---|---|---|---|---|---|---|
| 1536×4096 | 1 / 32 | 511 / 544 | 160 / 165 | **8.8 / 35.7** | 2.6e-2 | **1.7e-3** |
| 4096×1024 | 1 / 32 | 162 / 168 | 51 / 74 | **6.7 / 14.9** | 2.6e-2 | **1.7e-3** |
| 4096×256 | 1 / 32 | 75 / 80 | 20 / 26 | **5.9 / 7.0** | 2.6e-2 | **1.7e-3** |
| 512×4096 | 1 / 32 | 479 / 494 | 153 / 156 | **5.2 / 22.2** | 2.6e-2 | **1.7e-3** |
| 8192×1024 | 1 / 32 | 236 / 257 | 78 / 114 | **11.9 / 26.6** | 2.6e-2 | **1.7e-3** |
| one call per shape, M=1 | | 1463 | 463 | **39** | | |

- **Tuning helps ~3×** (best configs use BLOCK_M=16 and narrow N tiles, i.e. more programs;
  saved in `fp8_configs/` as vLLM config files), but the FP8 kernel stays 5–20× slower than
  bf16: gfx90a has no FP8 units, so every tile is converted in software.
- **bf16 is 20–90× faster than stock** through vLLM's own ROCm GEMM dispatch (`wvSplitK`
  skinny kernels at M ≤ 5, hipBLASLt above). Dequantising the weights is exact (e4m3 ×
  power-of-two scale fits bf16). Against an fp32 product it is also 15× more accurate,
  because W8A8 rounds the activations to FP8 and bf16 does not; but the model turns out to
  need that rounding (next section).
- Cost: the dense weights double, about +1 GB per GCD out of ~17 GiB of KV cache.

At M=1, 1463 µs per set of shapes × 43 layers ≈ 63 ms of the profiled 89 ms (some shapes run
more than once per layer); bf16 would take ~2 ms.

### bf16 dense linears (jobs 23535577, 23535685)

`BF16_LINEAR=1` (plugin `DSV4_BF16_LINEAR`, `bf16_linear.py`): after loading, every
block-FP8 dense linear is dequantised once to bf16 and served by vLLM's ROCm bf16 GEMM; at
load each layer is checked against the stock W8A8 apply on a random input. 256 layers per
rank, +0.96 GiB per GCD; the KV cache still holds 185k tokens (5.6× at 32k).

**Accuracy needs the activations rounded to FP8.** Feeding bf16 activations straight in
(job 23535577) matched W8A8 to its expected 2.6e-2 at load, but GSM8K fell to **90.5 %**
with 10/200 garbage answers, sanity 3/5. Not a GEMM bug: vLLM's skinny bf16 kernels give
the same 1.7e-3 error on strided and contiguous inputs (`test_skinny_gemm_strides.py`,
job 23535652). Rounding the activations to FP8 per 128-group with
fp32 scales first, as the stock path does, and then dequantising them (job 23535685: max
rel diff vs W8A8 2.7e-3 across all layers) restores **GSM8K 98.0 %, 0 garbage, sanity 5/5,
needles ok**. The model was trained with FP8 activations and depends on that rounding,
so it is now the default (`BF16_ACT=bf16` turns it off). It costs ~6 ms per step.

Recipe switches + `COMPILE=1 TRITON_LOGITS=1 NO_ASYNC=1 BF16_LINEAR=1`, piecewise graphs:

| concurrency | out tok/s | median TTFT | median TPOT | p99 TPOT | vs Triton logits | vs eager |
|---|---|---|---|---|---|---|
| 1 | 9.3 | 0.5 s | 107 ms | 107 ms | 1.75× | 3.1× |
| 4 | 34.2 | 1.7 s | 114 ms | 116 ms | 1.68× | 3.1× |
| 16 | 116.0 | 5.8 s | 126 ms | 137 ms | 1.63× | 3.1× |
| 32 | 190.0 | 7.8 s | 153 ms | 167 ms | 1.54× | 3.1× |

Without the activation rounding: 9.9 / 35.9 / 121.7 / 197.6 tok/s (TPOT 101/109/120/147 ms),
4–6 % faster but wrong. TPOT at 1 user fell 186 → 107 ms, close to the ~85 ms the profile
put on the FP8 kernel. What is left per step is mostly the hyper-connection fallback
(~54 ms), MoE (~18 ms) and attention (~18 ms).

### Hyper-connections: a correct pre kernel (job 23540893)

`TORCH_MHC=1` was a blunt fix: it turned off vLLM's whole fused TileLang mHC path, though only
one stage of one kernel is wrong. `plugin/lumi_dsv4_debug/mhc_fix.py` (`MHC_FIX=`, plugin
`DSV4_MHC_FIX`) keeps the fused path and replaces only `mhc_pre_big_fuse_tilelang`, the
kernel that turns the GEMM output into post/comb mixing weights (Sinkhorn) and the layer
input. vLLM's wrappers import it at call time, so patching the module attribute is enough.

That kernel is written for 32-wide warps: 96 threads per token, threads 0–31 run Sinkhorn
while threads 32–95 apply the pre weights. **The generated HIP source (TileLang's kernel
cache, `~/.tilelang/cache/0.1.10-x86_64/kernels/`) has no barrier at all inside that
branch.** Threads 32–35 write the pre weights to shared memory and threads 32–95 read them
with no `__syncthreads()`; each thread also reads residual values from shared memory that
other threads wrote, again without a barrier. Threads 32–63 and 64–95 are in different
wavefronts, so this is a cross-wave race. (On NVIDIA TileLang syncs such a thread subset with
a named barrier; HIP has none and the barrier is dropped.) Re-splitting it at 128 threads /
64 (`wave64`) puts the branch in one wavefront but emits a workgroup `__syncthreads()` that
only that wavefront reaches (undefined in HIP), and is still wrong in the test. A Triton
replacement (`triton`) is right.

`test_mhc_fix.py` (one GCD, real layer-5 parameters, vs the torch fallback that served V4
correctly), through vLLM's own `mhc_pre_tilelang` and `mhc_fused_post_pre_tilelang` wrappers:

| big-fuse kernel | layer_input rel err, T = 1 … 2048 | post/comb mixes, residual |
|---|---|---|
| stock TileLang | **0.7 – 1.8, inf** (wrong) | ok |
| TileLang at 128 threads split at 64 | **0.8 – 1.5, inf** (wrong) | ok |
| **Triton** | **1.5e-4 – 1.5e-3** | ok |

One decoder layer's hyper-connections (2 × pre + post), CUDA-graph replay, µs:

| tokens | torch fallback | fused, stock (wrong) | **fused, Triton** | per decode step (×43): torch → Triton |
|---|---|---|---|---|
| 1 | 710 | 54 | **37** | 30.5 → **1.6 ms** |
| 4 | 754 | 59 | **43** | 32.4 → **1.8 ms** |
| 16 | 841 | 86 | **71** | 36.1 → **3.0 ms** |
| 32 | 1029 | 96 | **84** | 44.3 → **3.6 ms** |
| 64 | 1120 | 135 | **134** | 48.2 → **5.8 ms** |

The torch fallback's ~30–48 ms per step (graph mode; ~54 ms eager in the profile) becomes
~2–4 ms, 10–20× less, and the Triton kernel is faster than the wrong TileLang one.

### Hyper-connections in serving: a NaN on the fused post path, and the variant that works

All three: recipe switches + `COMPILE=1 TRITON_LOGITS=1 NO_ASYNC=1 BF16_LINEAR=1`, piecewise
graphs, probe with 500 GSM8K questions (greedy, thinking off).

| hyper-connections | job | GSM8K (500) | runaway | translation probe | tok/s at 1 / 4 / 16 / 32 users | TPOT at 1 / 32 |
|---|---|---|---|---|---|---|
| torch fallback (`TORCH_MHC=1`) | 23541195, 23535685 | 98.0 % | 0 | ok | 9.3 / 34.2 / 116.0 / 190.0 | 107 / 153 ms |
| fully fused (`MHC_FIX=triton`) | 23541194, 23540919 | **96.2 %** | **5** | **NaN** | 14.3 / 52.4 / 176.2 / 282.4 | 69 / 103 ms |
| **torch post, TileLang + Triton pre** (`MHC_FIX=triton MHC_NATIVE=fusedpost`) | **23546991** | **97.6 %** | **0** | **ok** | **14.0 / 50.5 / 166.6 / 261.9** | **71 / 108 ms** |

97.6 vs 98.0 % is 2 of 500 questions (standard error ~0.6 points); the first 200 score
99.0 %. Sanity 5/5 and both needles pass. The last row is the recommended setup: **4.3–4.7×
eager, 1.4–1.5× the torch fallback**, and within 7 % of the fully fused path.

**The fully fused path is not usable**: with CUDA graphs, one probe prompt (a Finnish to
English translation, sanity item 5) gets a NaN logprob every time ("Out of range float values
are not JSON compliant: nan"), and GSM8K loses ~2 points with 5 runaway answers. Isolation:

| experiment | job | result |
|---|---|---|
| eager, every mHC call checked against torch | 23542348 | no mismatch, no NaN |
| eager, no checks | 23542631 | no NaN: graph mode only |
| `hc_head` on Triton instead of TileLang | 23542635 | NaN |
| GPU memory share 0.85 (KV cache 146k tokens, not 194k) | 23543941 | NaN |
| **post step in torch** (`MHC_NATIVE=fusedpost`) | 23542632 | **no NaN** |
| fused ops captured/replayed in graphs, whole 43-layer chain (`test_mhc_graph.py`) | 23543924 | bit-exact, no NaN |
| NaN/inf in graph padding rows | 23544718 | all ops row-local |
| argument layouts during capture (`MHC_META=1`) | 23544771 | all contiguous, offset 0 |
| per-site NaN tracer inside graphs (`MHC_NANTRACE=1`) | 23544961 | **NaN gone** |
| NaN-filled fresh allocations + tracer, fused and torch | 23545423, 23545424 | **NaN gone**, nothing flagged |

So it is tied to TileLang's fused post+GEMM step (`mhc_fused_tilelang` at ≤ 16 tokens,
`mhc_post_tilelang` + prenorm GEMM above), appears only under CUDA graphs, and disappears as
soon as any extra kernel runs between the mHC ops, which makes it untraceable this way. The
generated HIP sources of both kernels look correct (barriers where needed, launch shared-memory
sizes 96/128/10240 bytes matching their use, TileLang's warp reduction is 32 lanes wide by
design). Root cause not found; the `fusedpost` variant avoids it.

### Where decode time goes now, and three configuration experiments (Nov 9)

All with the recommended setup (`MHC_FIX=triton MHC_NATIVE=fusedpost TRITON_LOGITS=1
NO_ASYNC=1 BF16_LINEAR=1`), one change per allocation.

**Profile** (job 23547966, eager, `analyze_trace.py`, ms of GPU compute per decode step;
all-reduce left out, in an eager trace it is mostly ranks waiting for each other):

| component | 1 user | 32 users | note |
|---|---|---|---|
| MoE experts | 16.6 | 18.4 | at 1 user the MXFP4 matmul is only 24 %; `pack_bitmatrix` and other routing kernels most of the rest |
| sparse attention + other attention | 18.2 | 15.6 | `_sparse_attn_decode_ragged_kernel`, hipBLASLt |
| dense linears | 9.4 | 12.7 | bf16 GEMM ~3 ms; the FP8 activation rounding ~6 ms (quant 1.8 ms + a separate dequantise elementwise) |
| hyper-connections | 6.1 | 8.6 | Triton big-fuse, TileLang prenorm GEMM, torch post (hipBLASLt einsum) |
| decoder other, compressor, indexer, norms, head | 7.5 | 12.6 | |
| **GPU compute** | **~58** | **~68** | |
| time per token with graphs | 70 | 108 | |
| **not GPU compute** | **~12** | **~40** | grows with users |

Full decode graphs (below) remove the model's launch gaps and change nothing, so the
missing time is outside the model forward: RCCL (maybe 5–9 ms) and host work (scheduling,
input preparation, sampling), which `--no-async-scheduling` serialises with the GPU.

| experiment | job | tok/s at 1 / 4 / 16 / 32 users | accuracy | verdict |
|---|---|---|---|---|
| recommended setup | 23546991 | 14.0 / 50.5 / 166.6 / 261.9 | GSM8K 97.6 % (500) | |
| full decode graphs (`CUDAGRAPH_MODE=FULL_AND_PIECEWISE`) | 23547970 | 14.1 / 50.7 / 167.8 / 263.3 | 97.5 % (200) | no gain |
| expert parallelism (`--enable-expert-parallel`) | 23547968 | 12.8 / 49.4 / 160.1 / 250.6 | 97.0 % (200) | 4–9 % slower |

**More concurrent users** (job 23547967, `MORE_VLLM_ARGS="--max-num-seqs 128"`, KV cache
158k tokens instead of ~185k because more graph sizes are captured; accuracy not probed):

| users | out tok/s | per user | median TPOT | p99 TPOT |
|---|---|---|---|---|
| 32 | 259 | 8.1 | 108 ms | 122 ms |
| 64 | **376** | 5.9 | 155 ms | 167 ms |
| 128 | **503** | 3.9 | 238 ms | 248 ms |

1.45× the 32-user throughput at 64 users and 1.94× at 128; at 128 each user is below reading
speed, so 64 suits an interactive service and 128 batch work.

### Fused FP8 activation rounding; async scheduling; 64 sequences (Nov 9)

**Fused rounding** (`bf16_linear.fp8_round_triton`, now the default; `FP8_ROUND=torch` for
vLLM's quantiser + dequantise). `test_fp8_round.py` (job 23548678, one GCD): bit-identical
to the served path for every case (K 4096/1024/256, M 1–300, outlier, tiny, all-zero and
strided inputs), 2.3–4.9 µs per call instead of 11.5–14.3 µs. Served (job 23556297, 500
questions): **GSM8K 98.0 %**, differing from the recipe run on 4 questions, as two
torch-path runs differ from each other. Speed (job 23555546, same code):
**15.0 / 53.8 / 177.1 / 273.3 tok/s** at 1 / 4 / 16 / 32 users, TPOT 66 / 71 / 80 / 104 ms,
+4–7 % on the recipe.

**Async scheduling** (recipe without `NO_ASYNC=1`, job 23548655): no hang in a 17-minute
probe + sweep run (GSM8K 99 % of 100), but only +3.5–4 % (14.5 / 52.2 / 173.4 / 271.0 tok/s),
and vLLM's capacity estimate halves, 184k → 94k tokens, from the same 16.92 GiB. The cause:
`max_in_flight_tokens` = concurrent batches × `max_num_batched_tokens`, and every
sliding-window layer reserves window + in-flight tokens per request; with async there are 2
batches in flight, and V4's window is small, so the per-request reservation doubles. Not
adopted: little gain, and the reservation would cap admission at high concurrency.

**64 sequences** (`--max-num-seqs 64`, job 23555546): 390.5 tok/s at 64 users (TPOT 150 ms),
but **GSM8K 96.0 %**, losing 11 questions the recipe gets right (gaining 3): a regression,
not noise. With 64 sequences vLLM captures graphs up to 128 tokens instead of 64.

### A layout-sensitive bug under CUDA graphs (Nov 9)

`--max-num-seqs 64` reproduces it reliably, so it could be bisected. The GSM8K probe runs at
16 concurrent requests, so the batches themselves are the same at 32 or 64 sequences: only
buffer sizes and the CUDA-graph pool layout change. GSM8K over 500 questions, questions
lost / gained against the recipe run 23546991 (two clean runs differ by 1–3 questions):

| configuration | max seqs | job | GSM8K | lost / gained | runaway |
|---|---|---|---|---|---|
| torch hyper-connections (`TORCH_MHC=1`), torch baseline | 32 | 23541195 | 98.0 % | 3 / 1 | 0 |
| `MHC_FIX=triton MHC_NATIVE=fusedpost` (+ fused rounding) | 32 | 23546991, 23556297 | 97.6, 98.0 % | – / –, 1 / 3 | 0 |
| same, eager | 64 | 23557403 | 97.8 % | 2 / 3 | 0 |
| `TORCH_MHC=1` | 64 | 23557405 | 97.4 % | 3 / 2 | 0 |
| vLLM's fused decoder layer, every mHC op in torch (`MHC_NATIVE=fused,pre,post,head`) | 64 | 23559340 | 97.4 % | 2 / 1 | 0 |
| fusedpost | 64 | 23555546 | **96.0 %** | 11 / 3 | 0 |
| fusedpost, graph capture capped at 64 tokens | 64 | 23556639 | **96.0 %** | 8 / 0 | 5 |
| fusedpost, torch indexer logits (`TRITON_LOGITS=0`) | 64 | 23557404 | **95.8 %** | 13 / 4 | 0 |
| fusedpost, W8A8 FP8 linears (`BF16_LINEAR=0`) | 64 | 23557406 | **90.2 %** | 41 / 4 | 13 |
| TileLang-free (`fusedpost,gemm,post,head`) | 32 / 64 | 23558657, 23558655 | **96.4, 94.8 %** | 8 / 2, 18 / 4 | 7, 6 |
| vLLM's pre wrapper with a plain-torch big-fuse (`MHC_FIX=torch`) | 64 | 23560731 | **95.8 %** | 13 / 4 | 0 |
| no mHC custom op at all (`MHC_FIX=direct`: torch GEMM + Triton big-fuse) | 64 | 23561973 | **94.8 %** | 16 / 2 | 3 |
| same, full decode graphs (`FULL_AND_PIECEWISE`) | 64 | 23562860 | **93.8 %** | 20 / 1 | 1 |

What this rules out: my Triton big-fuse kernel (plain torch in its place still fails),
TileLang (the TileLang-free path fails), vLLM's mHC custom-op wrappers (the direct path
fails), vLLM's fused decoder layer (it passes with torch ops), the batched indexer kernel and
the bf16 linears (swapping them back still fails), breakable piecewise graphs (full decode
graphs fail too), and CUDA-graph replay of the mHC ops in isolation (`test_mhc_graph.py`:
bit-exact). Eager has never failed. Two implementations of the same pre-step arithmetic fail
while vLLM's `mhc_pre_torch` passes, and fusedpost passes at 32 sequences but fails at 64:
no single op explains it. It behaves like a memory-safety bug in something captured into CUDA
graphs whose effect depends on what sits where in the graph pool; the mHC code and the batch
limit only change that layout.

**Root cause: NaN pad rows reach the MoE router.** A graph is captured for a fixed number
of tokens, so a batch is padded up to the next capture size. Nothing initialises the pad
rows: they hold whatever is left in the graph pool, and that can be NaN. The rest follows
from two defects:

1. ROCm's V4 router (`ops.topk_hash_softplus_sqrt`, called by `fused_topk_bias`) returns ids
   `[0]*6` for an all-NaN row (`test_topk_nan.py`, job 23568164). An all-+Inf row gives
   `[0..5]` with NaN weights. Finite rows match the torch reference.
2. The `triton_unfused` experts build their dispatch and combine maps from a bitmatrix.
   A row with duplicate ids sets one bit for six top-k slots, so the maps fall out of step
   and **other rows** get the wrong expert outputs (`test_moe_pad_rows.py`, job 23569828).
   Padding R real rows to vLLM's capture sizes with `[0]*6` pad rows corrupts real rows at
   26 of 64 values of R. Examples (R, padded, rows corrupted): (17, 24, 10), (34, 40, 21),
   (61, 64, 10).

Eager runs have no pad rows, so they never fail. Whether a graph run fails depends on
whether the pool memory under the pad rows happens to hold NaN, which explains why the
results track memory layout and no single op.

**Fix:** `ROUTER_GUARD=1` (`router_guard.py`) replaces NaN in the gate logits with 0 before
top-k. That costs one elementwise op per MoE layer. Results, `MHC_FIX=direct` with 64
sequences (job 23568663, 500 questions):

| | GSM8K | lost / gained | runaway | tok/s at 1 / 32 / 64 users |
|---|---|---|---|---|
| without the guard (23561973) | 94.8 % | 16 / 2 | 3 | – |
| with the guard (23568663) | **97.2 %** | 4 / 2 | 1 | 14.3 / 261.3 / **377.9** |

TPOT was 69 / 107 / 154 ms. Two of the four lost questions, and the runaway (q119, which runs
to 1024 tokens), are also lost by the torch-mHC run at 64 sequences. The guard run reached
378 tok/s at 64 users, against 305 for the recipe.

**Caught in the act** (`GRAPH_DIFF=1`, no guard, job 23574590, 200 questions, 97.0 % with 3
garbage answers and 3 runaways). Each replayed step is rerun eagerly on the same padded inputs,
and every layer's outputs are compared. In steps padded to 120 tokens (113–118 real), the pad
rows were finite after the layer's attention but non-finite in its router logits. The MoE in
the same layer then made 3–5 *real* rows NaN, and the eager rerun of the same step did not.
This happened at layers 7, 21, 22, 23, 25 and 28. Pad rows also went non-finite first at
an attention output, so the garbage enters through more than one op working on
uninitialised pad memory. Which op it hits, and whether that op produces NaN, depends on
layout. That is why `TORCH_MHC=1` passed by luck.
Late in that run (from step ~1,700, while runaway requests were
decoding), attention in layers 29–33 also differed between graph and eager on real rows,
finite but with rel ≈ 1.

With the guard on (job 23586042, same setup), GSM8K was 97.5 % with no garbage and no
runaways. Pad rows still went non-finite 9 times in 1,500 steps, at an attention output or
at the router logits, but no real row diverged. The late attention divergence was fallout
from the earlier NaN, not a second bug. Only 4 divergences remain, at single-token steps
62–65 with rel 0.01–0.02. They appear in every diff run, with or without the guard, so they
are numerical noise in the rerun.

**Validated with the guard** (500 questions + speed sweep each; `ROUTER_GUARD=1` is now the
default in `run_v4_serve.sh`):

| configuration | max seqs | job | GSM8K | lost / gained | garbage / runaway | tok/s at 1 / 4 / 16 / 32 / 64 users | TPOT at 1 / 32 / 64 |
|---|---|---|---|---|---|---|---|
| `MHC_FIX=direct` | 32 | 23572794 | 97.4 % | 4 / 3 | 0 / 0 | 14.3 / 51.1 / 168.5 / 263.2 / – | 69 / 107 / – ms |
| `MHC_FIX=direct` | 64 | 23572795 | 97.8 % | 4 / 5 | 0 / 0 | 14.4 / 51.1 / 169.3 / 263.8 / 379.2 | 69 / 107 / 154 ms |
| `MHC_FIX=triton` (TileLang fused post, recipe) | 32 | 23591543 | 97.4 % | 3 / 2 | 0 / 0 | 15.5 / 56.1 / 187.0 / 296.1 / – | 64 / 97 / – ms |
| `MHC_FIX=triton` (TileLang fused post, recipe) | 64 | 23591544 | 97.8 % | 2 / 3 | 0 / 0 | 15.5 / 56.2 / 187.9 / 298.0 / **436.6** | 64 / 97 / 136 ms |
| `TORCH_MHC=1` | 64 | 23572796 | 98.0 % | 1 / 3 | 0 / 0 | 9.8 / – / – / 195.0 / 302.9 | 101 / 149 / 196 ms |

The fast path loses questions 93, 106 and 187 in both runs. In the probe these are near
misses, and the torch recipe at 64 sequences lost 93 as well. With the guard, the fast path
is 1.35–1.46× the torch recipe at 1–64 users and costs at most 0.6 GSM8K points.

### All-reduce: vLLM's custom kernel, made coherent on gfx90a (Nov 12–13)

A decode profile at 32 and 64 users (job 23631328, eager) adds up to ~60 and ~71 ms of
kernel time per step against 97 and 136 ms TPOT in graph mode; RCCL all-reduce is the
largest suspect for the gap (the eager trace can't measure it: ranks wait on each other).
vLLM's server log shows `Using ['PYNCCL'] all-reduce backends`: 0.26 enables its custom
all-reduce on ROCm only for gfx94x/95x (`RocmPlatform.use_custom_allreduce`). Its kernels
are in the gfx90a container, and amdsmi reports all 8 GCDs of a LUMI-G node one XGMI hop
apart, so vLLM's own topology check passes.

Forced on, it was 3–4.7× faster than RCCL in CUDA graphs, and the server reached 352 tok/s
at 32 users, but answered garbage (GSM8K 0 %, jobs 23671136/7). In graphs the one-stage
kernel reads each peer's input tensor over XGMI after a relaxed start barrier, and the
gfx90a L2 is per GCD and not coherent across XGMI. The peer's fresh input can still be
dirty in the peer's L2, and the reader keeps stale remote lines from the last step (graph
buffers reuse their addresses). My first test reduced the same input every time, so stale
data looked right. `test_allreduce.py` now writes new data before every replay
(job 23696578, 8 GCDs, bf16 [tokens, 4096], after a 16 MB write):

| tokens | RCCL | custom (stock) | + fence | + copy | max error: RCCL / stock / fence / copy |
|---|---|---|---|---|---|
| 1 | 37.0 µs | 9.9 µs | 12.7 µs | 12.2 µs | 0.09 / **17.3** / 0.03 / 0.03 |
| 32 | 82.9 | 24.5 | 26.7 | 25.2 | 0.11 / **13.6** / 0.03 / 0.03 |
| 64 | 110.7 | 31.5 | 33.6 | 32.7 | 0.12 / **14.7** / 0.03 / 0.03 |
| 128 | 122.5 | 51.1 | 53.6 | 53.8 | 0.15 / **18.9** / 0.03 / 0.03 |

Two fixes (`custom_ar.py`), both exact:

- `CUSTOM_AR=fence`: a one-thread kernel before each graph-captured call, with a
  system-scope acq_rel atomic, which compiles to `buffer_wbl2` + `buffer_invl2` on gfx90a.
- `CUSTOM_AR=copy` (= `1`): always copy the input into vLLM's IPC buffer, which ROCm builds
  already allocate uncached ("for signal on MI200"); vLLM skips that copy only in graphs.
  Simpler, and eager prefill can use it too, so it is the one in the recipe.

| | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 users | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| RCCL (recipe before) | 23591543/4 | 97.4 / 97.8 % | 15.5 / 56.1 / 187.5 / 297 / 436.6 | 64 / 69 / 77 / 97 / 136 ms |
| `CUSTOM_AR=fence` | 23696206/7 | 97.2 / 97.2 % | 16.3 / 62.3 / 192.4 / 318.3 / 465.4 | 61 / 62 / 75 / 90 / 127 ms |
| **`CUSTOM_AR=copy`** | 23696592/3 | **97.6 / 97.6 %** | 16.3 / 62.7 / 192.5 / **318.9** / **466.8** | 61 / 62 / 75 / 90 / 126 ms |

1.03–1.11× at every load (1.07–1.08× at 32–64 users). The stock kernel's 352/530 tok/s is not a ceiling: with garbage
hidden states the MoE routing degenerates. The gain matches the kernel numbers: ~58 µs less
per all-reduce at 32 tokens, ~86 per step.

With the custom all-reduce, at 128 users (500 GSM8K each; the server logs list `CUSTOM` for
the TP groups, while the job `.out` filters those lines out):

| layout | `--max-num-seqs` | job | GSM8K | tok/s at 1 / 16 / 32 / 64 / 128 users | TPOT at 1 / 32 / 64 / 128 |
|---|---|---|---|---|---|
| TP8 | 128 | 23700246 | 97.6 % | 16.3 / – / 315.6 / 465.2 / **626.8** | 61 / 90 / 126 / 191 ms |
| DP2×TP4 | 64 per replica | 23700244 | 97.0 % | 14.3 / – / 295.6 / **478.8** / **631.0** | 69 / 97 / 116 / 153 ms |
| DP2×TP4 | 32 per replica | 23700245 | 97.2 % | 14.2 / 172.3 / 312.6 / 384.9 / – | 70 / 87 / 115 / – ms |

TP8 gains 4 % at 128 users over RCCL (604 → 627 tok/s), but DP2×TP4 barely moves (476.5 →
478.8 at 64, 600 → 631 at 128) and stays slower per user (69 vs 61 ms at 1 user): without EP,
DP's MoE layers reduce over all 8 GCDs through the `ep` group, which keeps RCCL. TP8 stays the
recipe; DP2 is within 3 % at 64–128 users and not worth its latency.

### Compressor score GEMMs (Nov 13)

Each compressor layer computes `torch.mm(hidden, fused_wkv_wgate.weight.T,
out_dtype=torch.float32)`: bf16 [T, 4096] times a replicated bf16 [N, 4096] weight, N = 2048
(ratio-4), 1024 (ratio-128) or 512 (ratio-4 indexer), 60 calls per decode step. For an fp32
output hipBLASLt picks `Cijk_..._BSS_..._MT64x16x16_MI16x16x1`, i.e. fp32 MFMA on bf16
inputs: 66 µs per call at 32 users, 111 µs at 64, where the weight read takes ~10 µs.
`SCORE_GEMM=1` (`score_gemm.py`) replaces `attn_gemm_parallel_execute` (serial on ROCm
anyway) with a Triton kernel: bf16 MFMA, fp32 accumulation (bf16 products are exact in
fp32), split-K with a deterministic partial sum, tiles swept per N and batch; batches over
128 tokens (prefill) stay on `torch.mm`.

One GCD, graph replay, each call on a different weight copy (job 23706116); per decode step
(20 calls of each N):

| batch | 1 | 4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| `torch.mm` (ms) | 5.85 | 3.77 | 3.80 | 3.96 | 6.54 | 8.54 |
| Triton (ms) | 0.91 | 0.91 | 0.99 | 1.11 | 1.45 | 2.03 |

Max error against fp64 is 2–7e-7 relative, half that of `torch.mm`. In serving (jobs
23706308/9, against 23696592/3): TPOT at 1 user 61 → 56 ms (16.3 → 17.7 tok/s, +8 %),
192.5 → 198.2 tok/s at 16 users, 319 → 324 at 32 and 466.8 → 481.8 at 64 (+3 %; TPOT 126 →
122 ms). GSM8K 97.0 / 97.2 % (97.0–97.8 % across all recipe runs).

### Async scheduling, revisited (Nov 13–16)

On Nov 9 async scheduling gave +3.5–4 % but halved the KV capacity (above). vLLM sizes the
sliding-window reservation per request as window + `max_in_flight_tokens`, which is
concurrent batches × `max_num_batched_tokens` (`VllmConfig.max_in_flight_tokens`): 2 × 8192
with async, 1 × 8192 without. With `--max-num-batched-tokens 4096` async reserves what sync
did, and the capacity rises instead (jobs 23707179/80, against 23706308/9, same code):

| users | sync, 8192-token chunks | async, 4096-token chunks |
|---|---|---|
| 1 | 17.7 tok/s, TPOT 56 ms | **18.6**, 53 ms |
| 16 | 198.2, TTFT 4.2 s | **211.0**, 3.2 s |
| 32 | 324, TPOT 89 ms, TTFT 5.5 s | **349**, 83 ms, 3.6 s |
| 64 | 481.8, TPOT 122 ms, TTFT 5.3 s | **521.7**, 114 ms, 4.5 s |
| KV cache | 194k / 184k tokens | 289k / 280k tokens |

GSM8K 97.8 / 97.4 %, sanity 5/5, needles at 2.1k and 16.8k tokens; no hang through 500
questions and both sweeps (the earlier hang was before the indexer kernel's fixes). Prompts
over 4096 tokens now prefill in more chunks.

The control, 4096-token chunks without async (job 23782435, 64 sequences, GSM8K 97.8 %),
separates the two: the smaller chunk alone gives 17.6 / 200.7 / 333.5 / 493.6 tok/s at 1 /
16 / 32 / 64 users, a KV cache of 534k tokens and TTFT 2.5 / 2.7 / 3.6 s at 16 / 32 / 64
(smaller chunks also lower vLLM's profiled activation peak, and fewer prefills share a
step). Async then adds 5–6 % throughput and 3 ms of TPOT at 1 user, and costs half that KV
capacity and ~1 s of TTFT. **Async with 2048-token chunks** (job 23783289, 64 sequences,
GSM8K 97.8 %) has both: 18.7 / 210.7 / 350.4 / 523.0 tok/s at 1 / 16 / 32 / 64 users, KV
cache 625k tokens, TTFT 2.1 / 2.4 / 2.4 s at 16 / 32 / 64.

### MTP speculative decoding; split-K for small batches (Nov 16)

V4-Flash has one MTP layer and vLLM 0.26 has an AMD `DeepSeekV4MTP`. With one draft token
the sparse indexer gets two query tokens per sequence (`next_n` = 2, not flattened on this
GPU) with a causal length per token, `context_lens` [B, 2]. `paged_logits.py` now takes any
`next_n` (row r = b·next_n + j uses its own length and sequence b's block table);
`test_paged_logits.py` section 4 checks it against vLLM's `next_n` = 1 torch path per token:
rel err ~1e-7, same masks and top-512 (job 23783466). vLLM's own fallback for `next_n` > 1
cannot serve as the reference: it slices the cache as interleaved [D values, 4 scale bytes]
per position, while V4 writes per block (all values, then the scales), so it reads garbage.

**MTP, 1 draft token** (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`,
recipe with async and 4096-token chunks, 32 sequences, job 23783475): GSM8K 97.8 %,
sanity 5/5, needles 2/2, draft acceptance 59–73 % (mean acceptance length 1.6–1.7):

| users | 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| without MTP (23707179) | 18.6 tok/s, TPOT 53 ms | 68.2 | 211.0 | 348.9 |
| **MTP** | **31.6**, TPOT 32 ms | **104.0** | **310.8** | **436.0** |

1.70× per user at 1 user, still 1.25× at 32. The step costs about what a plain decode step
costs, so each accepted draft is nearly free.

**Split-K sparse decode for ≤ 4 queries** (`SPARSE_SPLIT=1 SPARSE_SPLIT_MAX=4`, jobs
23783284/5, async and 4096-token chunks, GSM8K 97.4 / 97.8 %): TPOT at 1 user 53 → 49.5 ms
(19.9 tok/s), 72.1 tok/s at 4 users (from 68.2), unchanged at 16–64 users. (One answer was
flagged as garbage by the probe; it is a correct answer in LaTeX.)

**Combined** (`MTP=1 SPARSE_SPLIT=1 SPARSE_SPLIT_MAX=8`, 2048-token chunks; with MTP each
sequence brings 2 queries, so 8 covers 1–4 users), and 2 draft tokens for comparison:

| | job | GSM8K | acceptance | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 | TTFT at 16 / 32 / 64 |
|---|---|---|---|---|---|---|
| no MTP (async, 2048 chunks) | 23783289 | 97.8 % | – | 18.7 / – / 210.7 / 350.4 / 523.0 | 53 ms | 2.1 / 2.4 / 2.4 s |
| **MTP 1**, 32 seqs | 23802055 | 97.6 % | 82 % | **34.3 / 118.9 / 307.2 / 446.2** / – | **29 ms** | 1.0 / 1.1 / – s |
| **MTP 1**, 64 seqs | 23802056 | 97.2 % | 75 % | **35.1** / – / 296.2 / 411.0 / **571.2** | 28 ms | 0.7 / 1.1 / 1.7 s |
| MTP 2, 32 seqs | 23802057 | 97.6 % | 53 % | **39.0** / 99.3 / 276.5 / 413.7 / – | 24.5 ms | 0.8 / 1.6 / – s |

Sanity 5/5 and needles 2/2 in all; KV cache 590–620k tokens. One draft token wins from 4
users up. A second draft is reused from the same single MTP layer and accepted only ~11 % of
the time at its position, so it pays only for a single user (39 tok/s). At 2 drafts
(`next_n` = 3) vLLM flattens the indexer queries to `next_n` = 1, which the kernel also
handles. Acceptance includes the GSM8K probe; free-form chat will accept less.

The recipe command run verbatim (job 23810538, `MTP=1` switch, concurrency 1 / 4 / 16 / 32 /
64): 34.8 / 114.0 / 275.2 / 431.2 / 630.3 tok/s, TTFT 0.3 / 0.6 / 1.0 / 1.6 / 2.1 s, GSM8K
97.4 %, acceptance 75 %. With MTP the per-load numbers spread more between runs than
without (16 users: 275–307 tok/s; 64 users: 571–630), as acceptance depends on the prompts.

### MoE at decode sizes (Nov 17)

An eager profile at 1 user without MTP (job 23782732, rank 0, ~43 ms of GPU work per step)
put the MoE at 16.5 ms, and most of it in kernels sized for prefill. Over the 43 MoE layers:

- **`pack_bitmatrix`**, 96 µs a call (4.1 ms). `make_routing_data` launches it with 512-row
  blocks whatever the batch, so one token runs as one workgroup sweeping [512, 32] tiles.
  16-row blocks give a bit-identical bitmatrix and are fastest at every size (graph replay,
  job 23811434): 43.7 → 3.3 µs at 1–512 rows, 43.6 → 6.7 µs at 8192.
- **Expert GEMMs** (`matmul_ogs`, MXFP4), 93 + 69 µs. triton_kernels' AMD opt flags pick
  block_n 256 for the gate/up GEMM (N = 512 per rank) and size split-K from a grid that
  counts all 256 experts. Forcing block_n / split_k per token count (`test_moe_matmul.py`,
  job 23811580, 8 routings cycled so the weights come from HBM): gate/up 37.6 → 23.6 µs at
  1 token, 59 → 37 at 2, 145 → 100 at 8, 430 → 274 at 32, 525 → 390 at 64; down 29.1 →
  17.8 µs at 1 and 34 → 28 at 2, already best from 4 tokens.
- **Router GEMM** (4096 → 256, fp32 out), 40 µs. `GateLinear` has no ROCm tier and falls
  through to bf16 `F.linear` (hipBLASLt with fp32 MFMA) plus a cast, so the logits are also
  rounded to bf16 (rel err 2e-3). `score_mm` does it in ~10 µs with fp32 output (2.5e-7).

`MOE_SMALL=1` (`moe_small.py`) wraps `pack_bitmatrix` and sets the opt-flag constraints
around each `matmul_ogs` call up to 128 tokens; larger batches keep the library's choice.
`GATE_GEMM=1` routes `GateLinear` through `score_mm`. With the rest of the recipe:

| | job | GSM8K | acceptance | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 16 / 32 / 64 | TTFT at 32 / 64 |
|---|---|---|---|---|---|---|
| recipe, 32 seqs | 23802055 | 97.6 % | 82 % | 34.2 / 118.9 / 307.1 / 446.2 / – | 28.6 / 44.4 / 58.9 / – ms | 3.1 / – s |
| **+ MoE**, 32 seqs | 23811671 | 97.4 % | 81 % | **35.2 / 122.1 / 333.6 / 501.9** / – | 27.6 / 41.5 / 54.0 / – ms | 1.0 / – s |
| recipe, 64 seqs | 23810538 | 97.4 % | 75 % | 34.8 / 114.0 / 275.2 / 431.2 / 630.3 | 28.1 / 46.9 / 60.2 / 85.1 ms | 3.2 / 5.7 s |
| **+ MoE**, 64 seqs | 23811670 | 97.6 % | 74 % | **35.4 / 128.0 / 306.2 / 464.9 / 643.9** | 28.1 / 41.1 / 53.9 / 79.9 ms | 1.3 / 1.7 s |

Sanity 5/5, needles 2/2, no garbage. TPOT drops 3–6 ms at 16–64 users and TTFT at 32–64
users by 2–3×, as prefill chunks route through the same kernels. At 1 user TPOT does not
move: the ~4 ms the eager profile promised does not appear under graphs, where these
kernels evidently cost less than their eager launches (not profiled yet).

### Where the 1-user step goes; full decode graphs (Nov 18–19)

The MoE fixes did not shorten the 1-user step, so I profiled the recipe with piecewise
graphs (`PROFILE=1 COMPILE=1`, jobs 23818676/7, all 8 ranks). `graph_trace.py` splits a
graph-mode trace into compute, all-reduce, and GPU idle time, and marks a gap as
host-late when the next kernel's launch call came after the GPU went idle:

| 1 user, per step (main model + MTP) | rank 4 | the other 7 ranks |
|---|---|---|
| GPU compute (busy minus all-reduce) | 36.7 ms | 37.2–37.5 ms |
| all-reduce kernels | 2.2 ms | 41–52 ms, spinning |
| host-late gaps | 51 ms (477 gaps) | 0.4–11 ms |

Every rank does the same ~37 ms of work. One rank's host launches nearly every kernel late,
and the custom all-reduce makes the other seven spin until it arrives. The slow rank is
different in each job (rank 2 at 32 users), and the profiler's own launch overhead
inflates it, but serving shows the same pattern: a ~50 ms step against ~39 ms of GPU work.

Two fixes, one allocation each, recipe at 64 sequences, with a same-day control (MTP runs
vary by 1–2 tok/s at 1 user between days):

| | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 16 / 32 / 64 |
|---|---|---|---|---|
| control | 23836296 | 97.2 % | 37.3 / 126.8 / 329.9 / 488.0 / 629.4 | 26.2 / 40.6 / 54.6 / 80.9 ms |
| `NUMA_BIND=1` (workers pinned to their GCD's CCD) | 23836294 | 97.6 % | 36.7 / 125.1 / 324.6 / 461.9 / 642.9 | 26.6 / 42.0 / 55.3 / 81.4 ms |
| **`CUDAGRAPH_MODE=FULL_AND_PIECEWISE`** | 23836295 | 97.2 % | **42.0** / 120.3 / 308.6 / 499.3 / 648.1 | **23.2** / 41.5 / 54.3 / 80.8 ms |
| same, 32 sequences | 23851784 | 97.4 % | **42.7** / 129.8 / 331.6 / 496.1 / – | **22.8** / 41.3 / 54.5 / – ms |

Full decode graphs (one graph launch per step for batches up to 2× the sequence limit,
which covers MTP's 2 tokens per sequence) give 1.13–1.15× at 1 user. From 4 users up, TPOT is
within 1 ms of the control; the 4- and 16-user throughput differences come from the few
prompts per level (8–32), not from slower decoding. Pinning workers changes nothing. On
Nov 9 full graphs gave nothing, when the GPU work was ~3× larger and hid the host.

### Sparse decode kernel from V4.1 (Nov 19)

With full graphs the 1-user step is GPU-bound, and sparse attention was its largest item
(~6.9 ms of ~37). The V4.1 port has its own gfx90a decode kernel (split-K partial + reduce,
16 KV entries per iteration, one scale load per token, NoPE and RoPE in one bf16 tile, the
main and compressed lists walked as one). V4-Flash shares the attention stack and the
584-byte `fp8_ds_mla` record, so the kernel is copied unchanged (`sparse_decode.py`, from
fd3fd13 on `exp/deepseek-v41-gfx90a`); only the hook differs, as stock 0.26's
`_rocm_sparse_attn_decode_ragged_triton` has no `out` argument. `SPARSE_DECODE=1`
supersedes `SPARSE_SPLIT`.

`test_sparse_decode.py` (job 23857050, one GCD, 8 heads): output within 2e-2 of the stock
fallback in all 60 cases (fixed and ragged lists, -1 slots, NaN in unused cache bytes, with
and without the sink, called with keywords through the hook). GPU µs per call:

| queries (MTP: 2 per user) | 2 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ratio 4 (top-512): fallback / split-K / **kernel** | 407 / 63 / **30** | 409 / 82 / **36** | 418 / 147 / **56** | 435 / 275 / **108** | 444 / 589 / **189** | 887 / 1171 / **338** |
| ratio 128: fallback / split-K / **kernel** | 130 / 61 / **15** | 129 / 61 / **18** | 130 / 62 / **27** | 132 / 89 / **62** | 140 / 233 / **88** | 281 / 461 / **129** |
| SWA only: fallback / split-K / **kernel** | 89 / 27 / **14** | 89 / 27 / **15** | 91 / 28 / **22** | 91 / 47 / **36** | 94 / 78 / **56** | 187 / 153 / **92** |

Serving, recipe with full graphs, same day:

| | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 16 / 32 / 64 |
|---|---|---|---|---|
| `SPARSE_SPLIT=1 SPARSE_SPLIT_MAX=8`, 64 seqs | 23857053 | 97.4 % | 42.0 / 134.8 / 324.9 / 468.7 / 649.1 | 23.2 / 40.8 / 54.5 / 80.8 ms |
| **`SPARSE_DECODE=1`**, 64 seqs | 23857052 | 97.4 % | **48.2 / 141.1 / 346.9 / 494.2 / 707.0** | **20.2 / 38.6 / 51.9 / 74.9 ms** |
| **`SPARSE_DECODE=1`**, 32 seqs | 23858696 | 97.2 % | **48.2 / 135.9 / 365.8 / 512.5** / – | **20.1 / 37.7 / 51.7** / – ms |

1.15× at 1 user, 1.05–1.09× from 4 to 64; TPOT 2–6 ms lower at every load. Sanity 5/5,
needles 2/2, no garbage.

### Dense bf16 GEMMs at decode sizes: `wo_a` and the linears (Nov 19–20)

Profiles of the recipe (`PROFILE=1`, jobs 23861347 at 1 user and 23861348 at 32) put the
dense bf16 GEMMs next after the MoE: hipBLASLt `Cijk_*` kernels took ~8 ms of a 54.5 ms
step at 32 users, and V4's output projection `wo_a` 1.9 ms of 33 at 1 user. vLLM's ROCm
dispatch (`rocm_unquantized_gemm_impl`) uses its skinny kernels (`wvSplitK`, `LLMM1`) only
up to 4–5 tokens. Above that, hipBLASLt takes nearly flat time from 8 to 128 rows: 33 µs for
the 1024 × 4096 `wo_a`. The `score_mm` Triton kernel (from `SCORE_GEMM`) with a bf16 output
beats it at almost every decode size:

- **`WO_A=1`** (`wo_a.py`): `wo_a` is a grouped einsum (`rocm_inv_rope_einsum`). At TP=8
  each rank holds one group, so it is a plain GEMM, which torch sends to hipBLASLt. The
  hook keeps the stock inverse RoPE and cached bf16 weight. It runs `wvSplitK` up to 3
  tokens (9–12 µs) and `score_mm` above that: 16 µs at 4–16 tokens, 24 at 64, 34 at 128
  (job 23866833). Output is within 2.4e-3 of the stock einsum (job 23866911). `score_mm`'s
  split-K sum is now a Triton kernel, 2–3 µs per call faster than torch's `.sum(0)` and equal
  to 1e-6; this also speeds up `SCORE_GEMM` and `GATE_GEMM`.
- **`DENSE_GEMM=1`** (`dense_gemm.py`): the same kernel for the other bf16 linears, by
  hooking vLLM's `rocm_unquantized_gemm`. Tiles come per shape and token count from a sweep
  against vLLM's own dispatch (`test_dense_gemm.py`, job 23880540). Where vLLM wins (2 tokens
  on three of the five shapes) the table keeps the stock path. Per rank, µs per call:

| shape (N × K per rank) | 8 tokens: vLLM / **Triton** | 64 tokens: vLLM / **Triton** |
|---|---|---|
| fused `wq_a` + `wkv` (1536 × 4096) | 35 / **15** | 37 / **25** |
| `wq_b`, `wo_b` (4096 × 1024) | 16 / **11** | 21 / **17** |
| shared expert w1+w3 (512 × 4096) | 33 / **9** | 32 / **12** |
| shared expert w2 (4096 × 256) | 9 / **5** | 8 / **6** |
| indexer `wq_b` (1024 × 1024) | 12 / **6** | 12 / **8** |

Summed over one layer's GEMMs: 55 → 50 µs at 2 tokens, 137 → 57 at 8, 139 → 66 at 32,
145 → 87 at 64, 157 → 125 at 128. Errors against fp64 are at bf16 rounding (2–4e-3) for both.

Serving, recipe + `WO_A=1`, one allocation each:

| | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| control (`WO_A=1`), 64 seqs | 23880590 | 98.0 % | 50.0 / 149.7 / 334.4 / 529.0 / 694.5 | 19.4 / 24.0 / 38.9 / 50.8 / 74.9 ms |
| **`DENSE_GEMM=1`**, 64 seqs | 23880589 | 97.0 % | 44.8 / **169.7 / 364.4** / 524.1 / **717.1** | 21.7 / **21.8 / 35.3 / 49.5 / 73.2** ms |
| `WO_A=1`, 32 seqs | 23880541 | 97.6 % | 54.8 / 154.2 / 346.3 / 500.6 / – | 17.6 / 23.9 / 37.8 / 50.9 / – ms |
| **`DENSE_GEMM=1`**, 32 seqs | 23896453 | 97.2 % | 51.8 / 141.3 / **360.9 / 538.7** / – | 18.1 / **21.7 / 34.7 / 49.0** / – ms |

`DENSE_GEMM` lowers TPOT by 1.3–3.6 ms from 4 users up, close to the ~2 ms per step the
sweep predicts. `WO_A`, against the Nov 19 runs without it (23857052, 23858696), is within
day-to-day noise from 16 users up and 0.5–2.5 ms faster at 1–4. The profile puts its gain at
~1 ms per step.

**The 1-user numbers here measure MTP acceptance as much as speed.** The benchmark prompts
are random tokens, so a tiny numerical change sends generation down a different path. The
server log shows `DENSE_GEMM`'s 1-user phase accepting 1.2–1.7 draft tokens per step against
1.5–1.7 in the control and ~1.8 in 23880541, at the same step rate. Compare 1-user runs by
TPOT × acceptance length, or with natural prompts.

### Hyper-connections from the V4.1 port; natural prompts (Nov 20)

With `MHC_FIX=triton`, a fused post+pre call is TileLang's post (18.7 µs at 64 tokens), its
prenorm GEMM (43.6 µs) and our big fuse (18.0 µs): 88 calls, ~7 ms of a 54.5 ms step at 32
users (job 23861348). Without DeepGEMM vLLM runs that GEMM with one K split, one workgroup
per token. The V4.1 port replaced the same work with Triton (`lumi_dsv41/mhc_triton.py`):
a post kernel and a split-K prenorm GEMM, per token or tiled with an fp32 matrix-core dot.
`mhc_fast.py` (`MHC_FAST=1`) brings both to V4. The partial sums go straight to the
big-fuse kernel, which already adds up any number of splits, so there is no reduce kernel.
A third variant, post and GEMM in one kernel, lost at every size and is unused.

`test_mhc_fast.py` (job 23898369, one GCD, layer-5 parameters): all variants within 1e-4 of
the torch references (served path: 2e-4 to 1.3e-3). One fused post+pre call, µs:

| tokens | 2 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| served (TileLang post + GEMM, Triton big fuse) | 19.2 | 28.9 | 35.3 | 41.6 | 65.5 | 114.6 |
| **`MHC_FAST`** | **14.6** | **17.0** | **21.0** | **25.1** | **34.7** | **44.2** |

At 2 tokens the big fuse is half of what is left (7.6 µs, ~2.5 µs of it the 20 Sinkhorn
iterations). Above 128 tokens (prefill chunks) vLLM's path stays.

**Natural prompts.** The sweep now has `BENCH_DATA=gsm8k` (from the V4.1 branch): GSM8K test
questions (~73 tokens) answered step by step up to 512 tokens, ending at EOS, 4 prompts per
user slot. MTP accepts 1.89–1.92 tokens per step there in every run, against 1.2–1.8 on random
tokens. These contexts are much shorter than the 1024-token random prompts, so the numbers
are higher and do not compare with the earlier rows. `PROFILE=1` uses the same prompts.

| 64 seqs, same day | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| control | 23898748 | 97.2 % (1 garbage) | 58.2 / 186.0 / 491.0 / 709.5 / 1068.9 | 16.7 / 19.4 / 27.2 / 35.6 / 52.1 ms |
| **`MHC_FAST=1`** | 23898747 | 97.4 % | 57.0 / 184.7 / 513.4 / 709.5 / 1054.5 | 16.7 / **19.0 / 26.6 / 34.4 / 51.5** ms |
| `MHC_FAST=1`, 32 seqs | 23898749 | 97.0 % | 54.3 / 181.5 / 485.8 / 618.5 / – | 16.7 / 19.2 / 27.7 / 41.7 / – ms |

TPOT 0.4–1.2 ms lower from 4 users up. Throughput moves ±4 % both ways because output
lengths differ between runs. That is ~2 ms per step at 32 users, near the 2.7 ms the
one-GCD timings predict. At 64 users it is ~1 ms against a predicted 6. With 32 sequences,
32 users ran 13 % slower than with 64. The graph sizes explain it: 32 sequences capture
piecewise graphs only to 128 tokens. A step with 32 users' 64 decode tokens and one new
prompt (~90 tokens with the chat template) is past that and runs eager. With 64 sequences
the limit is 256. Random 1024-token prompts never fit a graph when they arrive, so this did
not show before. `--max-num-seqs 64` is the setting for every load.

**Where the rest went: a cold instruction cache.** Graph profiles of this recipe (jobs
23900445/6, GSM8K prompts, all users decoding) show the post (6.2 µs) and prenorm GEMM
(19.1 µs at 32 users) at their one-GCD speed. The big fuse took 33 µs a call at both 32 and
64 users, against 12 µs alone; the pre-`MHC_FAST` profile had 18 against ~8. mhc_fix's
kernel unrolls the 20 Sinkhorn iterations and every split sum: ~3,400 lines of machine code,
run by one program per token. Alone, the code stays cached; in a decode step the kernels
in between evict it. `test_mhc_fast.py` part 4 (job 23900850) puts a GEMM, a softmax and a
top-k between calls. A whole fused post+pre call then costs 47 / 57 / 64 / 70 µs at 2 / 32 /
64 / 128 tokens with the unrolled kernel. With the same arithmetic in runtime loops (~1,400
lines, now the default; `MHC_BF=static` restores the old kernel) it costs 28 / 40 / 47 /
55 µs. Profiles with the loop kernel (jobs 23901341/2): **big fuse 14.1 µs at 32 users and
14.9 at 64** (from 33). GPU time per step falls 1.6 ms at 32 users and 1.9 at 64, so mHC is
3.5 ms a step at 32 users (7.1 before `MHC_FAST`). The serving A/B (23900889 loop vs
23900890 static, same day) is within run-to-run spread: TPOT 0.5–1.1 ms lower at 1–32 users,
0.5 higher at 64, GSM8K 97.4 vs 97.0 %. Any long unrolled Triton kernel served this way pays
the same, so time candidates with other kernels between calls.

| per decode step, GSM8K prompts | 32 users | 64 users |
|---|---|---|
| GPU busy (loop big fuse) | 55.6 ms | 69.4 ms |
| MoE expert GEMMs (`matmul_ogs`, gate/up + down) | 16.5 | 21.2 |
| Triton dense GEMMs (`score_mm`, 370 calls) | 7.1 | 9.9 |
| sparse attention (partial + reduce) | 4.7 | 6.5 |
| all-reduce (custom kernel, 90 calls) | 3.4 | 5.4 |
| mHC (prenorm GEMM + big fuse + post) | 3.5 | 4.3 |

### MXFP4 expert GEMMs: what bounds them, and a second tile table (Nov 21–22)

The two `matmul_ogs` GEMMs were 16.5 ms of a 55.6 ms step at 32 users. `test_moe_roofline.py`
(job 23908569, one GCD, 8 cold routings of uniform random experts) puts them against the
GCD's practical HBM read (1.25 TB/s, a sum over 512 MiB) and against the same GEMMs with
bf16 weights. **The FP4 → bf16 conversion bounds them, not memory.** gfx90a has no FP4 math,
so every weight is converted in software before the bf16 matrix cores. With bf16 weights the
GEMMs move 4× the bytes at 1.0–1.3 TB/s; with MXFP4 they reach 0.5–0.7 TB/s on the routed
experts' weights. The sweep also covers what the first table (`test_moe_matmul.py`) left
fixed: block_m, block_k, and num_warps / num_stages, which are overridden on the library's
returned flags. With a token or two per expert, block_m 16 instead of 32, 4 warps instead of
8, and one stage:

| µs per call | 2 tokens | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| gate/up, table v1 → **v2** | 37.5 → **30.1** | 98.7 → **84.9** | 260 → **220** | 390 → **326** | 469 → **391** |
| down, table v1 → **v2** | 27.8 → **21.1** | 64.3 → **49.9** | 163 → **126** | 235 → **178** | 293 → **225** |

Errors stay within bf16 rounding (≤ 3e-3). v2 is now `MOE_SMALL`'s table; `MOE_TABLE=v1`
keeps the old one. Graph profile at 32 users (job 23908712 vs 23901341): gate/up 223 →
214 µs, **down 153 → 130 µs**, GPU busy per step 55.6 → 54.3 ms. Gate/up gains less in
serving than in the test. Real routing reaches fewer experts than uniform: at 214 µs, 64
tokens behave like the test's 32 (~130 experts, not ~200).

| 64 seqs, GSM8K prompts, same day | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| `MOE_TABLE=v1` | 23908710 | 97.4 % | 58.7 / 202.1 / 530.7 / 727.3 / 1070.8 | 16.7 / 18.0 / 25.8 / 33.4 / 51.4 ms |
| **`MOE_TABLE=v2`** (recipe) | 23908709 | 97.0 % | **64.1 / 208.8 / 531.1 / 769.7 / 1090.1** | **14.9 / 17.9 / 25.0 / 32.7 / 49.6** ms |

TPOT is lower at every load, by 0.1–1.8 ms. Sanity 5/5, needles 2/2, no garbage in either
run. MTP acceptance is 1.91 vs 1.88.

What is left is the conversion. Reading the ~130 routed experts' weights at 1.2 TB/s would
take ~120 µs for gate/up and ~60 µs for down, against 214 and 130. That is ~7 ms a step at
32 users, the largest single item now. It needs a cheaper e2m1 → bf16 upcast in the kernel, not tiles.

Triton's own upcast is not easy to beat. `test_fp4_upcast.py` (job 23973504) runs one
decode-shaped GEMM (16 rows, K 4096, N 65536, ~128 experts' gate/up) four ways:
`tl.dot_scaled` as `matmul_ogs` calls it; nibbles placed into an fp32 bit pattern and scaled
with one multiply; the bf16 pattern built with integer ops; and bf16 weights. `dot_scaled`
was the fastest FP4 variant at every tile: 278 µs at block_n 256 / 8 warps, against 327 µs
for fp32 bits and 483–634 µs for integer bits. The fp32 variant also used a wrong scale
exponent; its error is 1.0, but it was slower anyway. With bf16 weights this simple kernel
reaches only 0.3–0.4 TB/s, against 1.0–1.3 for `matmul_ogs`, so it cannot show the
conversion's share by itself. The evidence that the conversion is the bound stays the
`matmul_ogs` comparison above. A budget estimate fits it: 0.65 TB/s of MXFP4 is 1.3 G weights a second. A GCD issues ~12 T
VALU lane-ops a second, so that is ~9 operations per weight for unpacking, scaling and
converting. Doubling the rate would mean ≤ 5 operations per weight: packed 16-bit ops or
`v_perm_b32` byte lookups through inline assembly. That is a hand-tuned kernel, not a flag.

### Dense bf16 GEMMs against the memory bound (Nov 23)

`score_mm` runs every bf16 dense GEMM at 4–128 tokens: 370 calls, 7.1 ms of a 32-user step.
At 0.2–0.85 TB/s on their weights they looked far from the 1.25 TB/s the GCD streams.
`test_dense_bw.py` (job 23973727, cold weights) re-swept all ten per-rank shapes at 8 / 32 /
64 / 128 tokens. It added what earlier sweeps held fixed: grid order (`M_FAST`, a weight
tile's token blocks side by side so they share L2), and warps × stages. **The served tiles
are within 3 % of the best.** Per decode step, weighted by calls: 3.97 → 3.91 ms at 8
tokens, 4.56 → 4.54 at 32, 6.04 → 5.93 at 64, 8.67 → 8.39 at 128. Grid order changes
nothing: L2 already catches the re-reads. The tables stay as they are; `score_mm` keeps the
optional `(…, num_stages, m_fast)` config fields.

The gap is not tiling. The weights are small (2–17 MB a call), so fixed costs dominate: the
launch, the split-K partial pass plus its sum kernel (238 calls, 0.82 ms a step), and few
programs per call. At 128 tokens there is also MFMA time. The bytes per decode step (~2.7 GB at
64 tokens) would take ~2.1 ms at 1.25 TB/s, against ~6 ms. Reaching that needs fewer,
larger calls: GEMMs that share an input fused into one (wq_a+wkv with the compressor score
GEMMs, the shared expert's w1+w3 with the router), or a split-K that reduces in-kernel.

### Small kernels around the MoE (Nov 23–24)

At 32 users a graph-replayed step held ~700 launches of small torch kernels:

| kernel | launches a step | ms a step |
|---|---|---|
| strided copies | 140 | 1.75 |
| elementwise | 142 | 1.43 |
| FP8 rounding | 256 | 0.99 |
| split-K sums | 252 | 0.86 |
| buffer copies | 188 | 0.61 |
| about ten per-layer fills, adds, casts and sigmoids | 46 each | ~2 |

A replayed kernel carries no Python stack, so `small_kernels.py` reads an eager profile
(`CUDAGRAPH_MODE=NONE`, job 23974135; CPU job 23983635) to find what launches them. Most
come from vLLM's MoE glue for the unfused MXFP4 experts. `make_routing_data` runs int16 and
bf16 casts, a zero fill, `pack_bitmatrix`, `ids == -1` and a `where`: six launches per layer,
most on torch's non-vectorised path. `UnfusedOAITritonExperts.apply` and the `gate_scal`
gather index with int32 tensors, which torch first casts to int64 in a strided copy. The
shared expert's `SiluAndMulWithClamp` is sent to `forward_native` on ROCm: two clamps, two
muls, a sigmoid and an add per layer.

`MOE_GLUE=1` (`moe_glue.py`) replaces them:
- one Triton kernel for the routing prologue: the int16 ids, the bf16 weights (−1 for empty
  slots) and the bitmatrix;
- a Triton row gather that takes int32 indices directly;
- one kernel for the activation, with `forward_native`'s bf16 rounding after each op.

`test_moe_glue.py` (jobs 23983883, 23983907): routing outputs equal to vLLM's on every
routed entry. With −1 slots vLLM's own padding differs between two runs on the same input.
Gathers and the activation are bit-identical. The replaced glue takes 101–116 → 70–83 µs per
MoE layer at 2–128 tokens, 1.2–1.6 ms a step.

| 64 seqs, GSM8K prompts, same day | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| control | 23983909 | 97.0 % | 65.1 / 198.5 / 543.4 / 750.7 / 1067.9 | 14.9 / 18.1 / 25.0 / 32.8 / 51.3 ms |
| **`MOE_GLUE=1`** | 23983908 | 97.4 % | **70.7 / 210.3 / 576.2 / 783.5 / 1098.2** | **13.6 / 17.1 / 23.7 / 31.9 / 50.1** ms |

TPOT is 0.9–1.3 ms lower at every load, 9 % at one user. Sanity 5/5, needles 2/2, no
garbage, MTP acceptance 1.91 vs 1.92. The one-user runs generate the same 653 tokens.

Two things tried here did not help. Fusing BF16_LINEAR's FP8 activation rounding into the
GEMM's tile load (`FP8_FUSE=1`, `test_fp8_fuse.py`, job 23974177) is bit-identical but
0.4–8 µs slower per call. Every program re-rounds the same activation tile, which costs
more than the launch it saves; it stays off. The buffer copies are mostly needed: the custom
all-reduce's copy into its uncached buffer, and the sparse decode and MoE finalize output
copies (~0.25 ms together).

### The routing prologue on the legacy path; q-norm / RoPE / KV insert (Nov 24)

A graph profile with `MOE_GLUE` (job 23988423) still showed the six routing-prologue
kernels. This container's `triton_kernels` has no `SparseMatrix`, so vLLM takes its legacy
path (`routing_from_bitmatrix`), and `moe_glue.py` had replaced only the other one. Its test
had compared vLLM's routing with itself; it now prints the path taken. The one-kernel
prologue now feeds the legacy `routing_from_bitmatrix` too (`MOE_GLUE_ROUTE=1`, the
default). `test_moe_glue.py` (job 23989302, legacy path): routing equal on every routed
entry, the replaced glue **100–117 → 19–22 µs per MoE layer**. vLLM's prologue alone was
~50 µs a layer.

The same profiles showed vLLM's HIP `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
taking 32–34 µs a call at 1, 32 and 64 users: ~1.4 ms a step. `qnorm_rope.py`
(`QNORM_ROPE=1`) does it in one Triton kernel: per-head RMSNorm and GPT-J RoPE into the
head-padded q, zeroed padding heads, and the KV row's RoPE, UE8M0 FP8 quantisation and
cache insert as vLLM's Triton `quantize_and_insert_k_kernel` writes it. `test_qnorm_rope.py`
(job 23988389): cache bytes identical, q identical but for one bf16 rounding at 64 tokens.
Alone the HIP op takes only 9–11 µs and the Triton kernel 6–12. In a decode step (profile
23988423) the Triton kernel takes 12.6 µs against the HIP op's 33.

| 64 seqs, GSM8K prompts, same day | job | GSM8K | tok/s at 1 / 4 / 16 / 32 / 64 | TPOT at 1 / 4 / 16 / 32 / 64 |
|---|---|---|---|---|
| `MOE_GLUE` without the legacy-path prologue | 23989318 | 97.6 % | 63.2 / 206.7 / 551.5 / 778.9 / 1165.0 | 13.6 / 16.8 / 23.8 / 31.5 / 48.9 ms |
| + legacy-path prologue (`MOE_GLUE_ROUTE=1`) | 23989319 | 97.2 % | 75.0 / 221.6 / 562.7 / 811.9 / 1135.7 | 12.8 / 16.0 / 22.9 / 30.9 / 47.6 ms |
| **+ `QNORM_ROPE=1`** (recipe) | 23989320 | 97.6 % | **75.6 / 223.4 / 603.2 / 824.1 / 1144.3** | **12.8 / 15.8 / 22.7 / 30.9 / 47.1** ms |

The prologue lowers TPOT by 0.6–1.3 ms at every load. `QNORM_ROPE` adds 0–0.5 ms, about the
0.5 ms its ~0.9 ms a step predicts at 1.9 tokens a step. Sanity 5/5 and needles 2/2 in all
three runs. The one "garbage" flag in 23989320 is a correct answer (15). The same question
was flagged in control 23898748: the checker reacts to its formatting.

### Prefill (Nov 25–26)

A profile of one 8,275-token prefill on the recipe (`PROFILE_PREFILL=8192`, job 24005447:
2.78 s, ~2,970 tokens/s) splits it into:

| component | share | main cost |
|---|---|---|
| sparse attention | 23 % | `_sparse_attn_prefill_ragged_kernel`, 2.8 ms a call |
| sparse indexer | 18 % | vLLM's torch fallback for the prefill logits (elementwise passes, 0.2–0.6 ms each) |
| hyper-connections | 14 % | TileLang prenorm GEMM, unrolled big fuse, TileLang post |
| attention (other) | 13 % | the compressor score GEMMs on hipBLASLt's fp32-MFMA kernel, 1.4 ms a call |
| MoE experts | 12.5 % | `matmul_ogs` on the library's default tile, 0.73 ms a call |
| dense GEMMs / all-reduce | 7 / 6 % | hipBLASLt; RCCL, mostly waiting for the slowest rank |

(A 16k profile's trace was cut off when the job ran out of time while writing it; 8k keeps
the same 2048-token chunks.)

- **`SPARSE_PREFILL=1`** (`sparse_prefill.py`): the V4.1 branch's text patch of 0.26's
  `_rocm_sparse_attn_prefill_ragged_triton` launch. 2 warps, one stage, `matrix_instr_nonkdim=16`,
  `waves_per_eu=1`, same tile and output (V4.1: 5.99 → 1.77 ms a chunk at 16k).
- **Score GEMMs at prefill sizes**: `score_mm` had tiles only up to 128 tokens. Above that,
  the compressor score GEMMs and the router fell back to `torch.mm`.
  `test_score_prefill.py` (job 24014949): 3.3–6.2× faster at 256–2048 tokens (N 2048: 2.40
  → 0.38 ms at 2048), error against fp64 equal or lower. On by default; `SCORE_PREFILL=0`
  turns them off.

16k-token random prompts, 16 output tokens, time to the first token (one allocation each):

| | job | 1 user | 4 users |
|---|---|---|---|
| recipe before | 24014951 | 6.45 s | 18.7 s |
| `SPARSE_PREFILL=1` | 24014950 | 5.51 s | 16.0 s |
| `SPARSE_PREFILL=1`, score tiles off (same day as the next row) | 24019176 | 5.53 s | 16.1 s |
| `SPARSE_PREFILL=1` + score tiles | 24019175 | 4.97 s | 14.6 s |
| the same (control, same day as the next row) | 24049247 | 4.98 s | 14.7 s |
| + `PREFILL_LOGITS=1` | 24049246 | 3.47 s | 9.38 s |
| the same (control, same day as the next row) | 24068559 | 3.48 s | 9.37 s |
| **+ `MHC_FAST` at prefill sizes** | 24068558 | **3.06 s** | **8.21 s** |

The first two together: −23 %. GSM8K 97.2 / 97.4 %, sanity 5/5 and both needles (the ~18k one included) in
the runs with them. Decode on GSM8K prompts with both (job 24019177): TPOT 12.5 / 16.6 / 23.0
/ 30.0 / 48.9 ms at 1 / 4 / 16 / 32 / 64 users, within earlier runs' spread. 

- **`PREFILL_LOGITS=1`** (`mqa_logits.py`, the V4.1 branch's kernel, with
  `DSV4_INDEXER_GATE`): the indexer's prefill logits in one Triton kernel. It replaces
  vLLM's torch fallback (`fp8_mqa_logits_torch`: a dense [heads, rows, keys] fp32 einsum,
  elementwise passes and a sum), which the plugin had split into row chunks to fit in memory.
  A program owns a logits tile, loads its keys once and loops over V4's 64 heads with bf16
  dots. It writes −inf without computing for tiles outside every row's causal range.
  `test_mqa_logits.py` (job 24049231, 2048 rows): same mask, max rel err 3e-7, top-512
  overlap 100 %; 9.1 → 0.65 ms at 2k keys, 35.8 → 3.6 at 8k, 75.6 → 7.7 at 16k. In
  serving, −30 % at one user and −36 % at four. GSM8K 97.0 %, sanity 5/5, needles 2/2.
  V4 now prefills 16k tokens in less time than the V4.1 port (3.77 s).

- **`MHC_FAST` at prefill sizes** (`MHC_PREFILL=1`, the default; `=0` leaves chunks above
  128 tokens on vLLM's path): the tiled prenorm GEMM (64 × 64 tiles, 512-wide K chunks) and
  the loop big fuse on one warp, now up to 2048 tokens. `test_mhc_fast.py` (job 24068455):
  one fused post+pre call 191 → 69 µs at 256 tokens, 360 → 123 at 512, 550 → 231 at 1024,
  1034 → 475 at 2048. Layer-input error against the torch reference is 2.5–4e-5, against
  1.7e-4 for the served path. In serving, −12 % at one and four users. GSM8K 97.4 %, sanity
  5/5, needles 2/2 (one "garbage" flag: the known false positive on a correct LaTeX answer).

- **MoE tiles at prefill sizes** (`MOE_PREFILL=1`, the default; `MOE_SMALL`'s table to 2048
  tokens): `test_moe_roofline.py` with block_m up to 128 (job 24069742) finds gate/up 712 →
  516 µs at 256 tokens and 929 → 828 at 2048, down 336 → 297 and 744 → 632, bit-identical.
  The GEMMs still read their weights at only 0.2–0.5 TB/s: prefill has the same FP4-conversion
  bound as decode. In serving it is within noise: 3.08 → 3.06 s at one user, 8.34 → 8.26 s at
  four (jobs 24070076 / 24070075, GSM8K 97.2 %). It stays on, as it is never slower per call.

Together −53 % since the prefill profile: 6.45 → 3.06 s for 16k tokens at one user, 18.7 →
8.2 s at four, ahead of the V4.1 port's 3.77 s. What is left has no single cheap fix: sparse
attention, the MoE FP4 conversion, dense GEMMs and the all-reduce (mostly ranks waiting).

### Prefix caching (Nov 27)

An agent or a chat resends its growing context every turn, so without prefix caching every
turn prefills it again: 3.2 s at 17k tokens, and under load the queue of such prefills
dominates. `run_v4_serve.sh` passed `--no-enable-prefix-caching`. The V4.1 port had to
change its own plugin to allow caching. V4, on vLLM's own model, needs no change. It only
needs the V4.1 port's retention setting, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: with 0.26's
dense retention, every sliding-window block is hashed and queued for reuse, and V4.1 found
cached prefixes evicted after a few hundred thousand tokens of other prefill.
`PREFIX_CACHING=1` adds `--enable-prefix-caching --enable-prompt-tokens-details` and sets the
interval to 4096 (`RETENTION`, `=dense` for 0.26's). The V4.1 branch's checks run as they
are: `PREFIX_CHECK=1` (`prefix_cache_check.py`), `AGENTIC=1` (`agentic_bench.py`) and
`PREFIX_EVICT=1` (`prefix_evict_probe.py`).

**Hit against cold** (`PREFIX_CHECK=1`, job 24087028, 24 few-shot prompts of ~17.3k tokens,
each sent cold twice and then as a hit):

| | cold | hit |
|---|---|---|
| time to first token | 3.2–4.5 s | **0.12–0.32 s** (16,384–17,408 tokens from the cache) |
| GSM8K answers right | 24 / 24 | 24 / 24 |
| first token's top-5 logprobs, mean gap to a cold run | 0.150 (cold to cold) | 0.148 |
| tokens shared with a cold run from the start, mean | 42.7 (cold to cold) | 54.9 |
| MTP acceptance length | 1.96 | 1.95 |

A hit is as close to a cold run as two cold runs are to each other. Greedy output is not
repeatable run to run here, so hits are judged against that spread. Agentic turns (2 ×
6, each resending a ~9k-token context with the previous answers): every turn after the
first reuses all but 200–700 tokens, 0.18–0.52 s to the first token against 1.75 s cold.
11/12 right; the miss is a last turn whose cold re-run also diverged after 59 of 151
tokens. GSM8K probe 97.5 %, sanity 5/5, needles 2/2.

**Agentic load** (`AGENTIC=1`, jobs 24087235 / 24087236): 32, then 64 conversations at once,
each resending a shared preamble, a ~8k-token document, the turns so far and a new GSM8K
question (≤ 384 output tokens), growing by ~1.5k tokens a turn to 30k. 15-minute windows
after 5 minutes of warm-up.

| | no prefix caching (24087236) | **prefix caching** (24087235) |
|---|---|---|
| 32 conversations: output tok/s | 24 | **174** |
| turns completed | 213 | 1610 |
| prompt tokens from the cache | 0 % | 87 % |
| TTFT median, 16–24k / 24–32k context | 99 / 141 s | **1.7 / 2.0 s** |
| TPOT median | 201–207 ms | 152–169 ms |
| 64 conversations: output tok/s | 30 | **180** |
| TTFT median, 8–16k / 16–24k | 136 / 194 s | **19 / 7.5 s** |
| TPOT median | 196–200 ms | 205–227 ms |
| answers right | 95 % | 95 % |

7.3× the output at 32 conversations, 6.0× at 64; no preemptions, no errors (the control's one
traceback is the engine stopping at the end of the job). The V4.1 port measured 22 → 144
and 27 → 154 tok/s under the same load. At 64 conversations TTFT is still 7–19 s:
every turn adds ~1.5k new tokens to prefill and the 2048-token chunks queue behind
each other.

Benchmarks: the `BENCH=1` sweep's prompts are unique, so with caching on they measure the
same as without. Fixed-seed random prompts that repeat across phases would be served from
the cache, so random-prompt sweeps should run without it.

### From the V4.1 branch: blocked FP8 rounding, larger custom all-reduce messages (Nov 24)

Two changes the V4.1 port made to this plugin, measured on V4:
- `FP8_ROUND_BLOCK` (default 2048): the FP8 activation rounding with many 128-column
  groups per program. `test_fp8_round.py` (job 23990224) finds it bit-identical. It takes
  3.5 → 2.8 µs a call at 32 rows and 92 → 35 µs at 2048 rows (K 4096).
- `CUSTOM_AR_MAX_MB`: the custom all-reduce's message limit, 8 MB in vLLM. V4's 2048-token
  prefill chunks make 16 MB messages, which otherwise go to RCCL.

Same day, both on (`CUSTOM_AR_MAX_MB=32`) against both off. Decode, 64 seqs, GSM8K
prompts: TPOT 12.8 / 15.7 / 22.8 / 30.0 / 48.9 ms against 12.6 / 15.8 / 22.4 / 30.4 /
46.2 at 1 / 4 / 16 / 32 / 64 users (jobs 23990225 / 23990227), GSM8K 97.8 / 97.6 %, all
within run spread. Prefill of 16k random-token prompts: time to first token 6.6 s against
6.5 s at one user, 19.0 against 18.8 s at four (23990226 / 23990228). Neither helps V4.
V4.1's gain came from its 32-column MXFP8 groups, where one program per group was mostly
launch overhead; V4's 128-column groups were not. The blocked kernel stays the default
(bit-identical, never slower per call); `CUSTOM_AR_MAX_MB` stays unset.

The same runs give the prefill baseline: **6.5 s for a 16k-token prompt** (~2.5k tokens/s),
where the V4.1 port, a larger model, takes 3.77 s after its prefill work.

### Split-K sparse attention decode; DP2×TP4; 128 sequences (Nov 11–12)

- **`SPARSE_SPLIT=1`** (`sparse_split.py`): vLLM's split-K decode kernels are gated to
  gfx942/950; gfx90a gets one workgroup per query. On one GCD (graph replay, job 23638388)
  split-K is 2–6.7× faster for ratio-4 layers at 1–16 queries (409 → 61 µs at 1) but 1.3×
  slower at 64+, so the plugin splits only batches of at most 16. In serving it cuts TPOT at
  1 user from 64 to 60 ms in every comparison, but same-day A/B runs (23670794/5) show
  +3 ms at 16 and 32 users, and a cutoff of 32 lost 10 ms at 32 users (23638474/5).
  Off by default; unexplained.
- **DP2×TP4** (64 sequences per replica, 200 GSM8K: 96.5 %, job 23598815): 14.5 / 309.5 /
  476.5 / 600.1 tok/s at 1 / 32 / 64 / 128 users, TPOT 69 / 89 / 114 / 158 ms. Without EP,
  vLLM shards the experts across all 8 GCDs, so it fits; 165k KV tokens per replica.
- **128 sequences** (TP=8, job 23598814, 97.0 %): 604.4 tok/s at 128 users, TPOT 199 ms.
- **Decode profile at 32 / 64 users** (eager, ms per step; job 23631328): MoE 18.4 / 21.2
  (`matmul_ogs` ~50 %, `pack_bitmatrix` ~20 %), sparse attention 9.7 / 9.9, bf16 GEMM
  6.7 / 7.6, attention other 5.6 / 8.3 (mostly the compressors' fp32-output score GEMM,
  hipBLASLt `MT64x16x16`), mHC 5.3 / 7.2, compressor 4.5 / 6.8.

## Open

- **Speed**, in likely order of payoff:
  1. ~~Graphs~~: done, piecewise, 1.3–1.7× (above).
  2. ~~Batched indexer logits~~: done, 2.0× eager at 32 users (above). FULL decode graphs
     add nothing. Open: the race with async scheduling.
  3. ~~Dense FP8 linears~~: done, bf16 GEMM with FP8-rounded activations, 1.5–1.75× on top
     (above). The rounding could be fused into a kernel to win back its ~6 ms per step.
  4. ~~A correct hyper-connection pre kernel~~: Triton big-fuse kernel (above). Served as
     `MHC_FIX=triton` + `ROUTER_GUARD=1`, 296 tok/s at 32 users and 437 at 64 (recipe).
     The NaN seen earlier with TileLang's fused post+GEMM was the pad-row bug: with the guard
     it passes (jobs 23591543, 23591544).
  5. ~~MXFP4 MoE at decode sizes~~: done, `MOE_SMALL=1 GATE_GEMM=1`, 1.03–1.12× at 4–32
     users (above). The ~15 small routing kernels around it remain.
  6. ~~Async scheduling~~: done, with 4096-token chunks, 1.05–1.08× (above).
  7. ~~Fuse the FP8 activation rounding~~: done Nov 9 (`fp8_round_triton`, the default).
  8. ~~MTP speculative decoding~~: done, `MTP=1`, 34–35 tok/s at 1 user, 1.09–1.8× (above).
  9. ~~All-reduce~~: done, vLLM's custom all-reduce with an uncached copy, 1.03–1.11×.
  10. ~~The compressors' fp32-output score GEMMs~~: done, Triton, 1.01–1.08× (above).
  11. ~~Host launch overhead at 1 user~~: done, full decode graphs, 1.13× (above). CPU
      pinning (`NUMA_BIND=1`) does not help.
  12. ~~Sparse attention decode~~: done, the V4.1 port's kernel (`SPARSE_DECODE=1`),
      1.05–1.15× (above).
  13. ~~Dense bf16 GEMMs at 4–128 tokens~~: done, `WO_A=1 DENSE_GEMM=1`, TPOT 1.3–3.6 ms
      lower from 4 users up (above). Left at 32 users: MXFP4 expert GEMMs (`matmul_ogs`,
      ~10 ms of 54), mHC `hc_prenorm` (3.8 ms), all-reduce (3.8 ms).
  14. ~~Hyper-connections at decode sizes~~: done, `MHC_FAST=1` (V4.1's Triton post and
      split-K prenorm GEMM), 1.3–2.6× per call, TPOT 0.4–1.2 ms lower from 4 users up;
      big fuse with runtime loops (cold instruction cache), mHC 7.1 → 3.5 ms a step at 32
      users. Next by size: MoE expert GEMMs (~30 % of the step), the 370 dense-GEMM calls,
      sparse attention, all-reduce.
  16. ~~Small kernels around the MoE~~: done, `MOE_GLUE=1` (routing prologue on both
      `triton_kernels` paths), TPOT 1.7–2.4 ms lower at every load; `QNORM_ROPE=1`, 0–0.5 more. Left: the split-K sum kernels (252 a step, 0.86 ms), FP8 rounding (0.99 ms; fusing
      it was slower), the routing metadata kernels.
  15. ~~Expert GEMM tiles, second pass~~: done, `MOE_SMALL` table v2 (block_m 16, 4 warps,
      one stage), down −15 % in serving, TPOT 0.1–1.8 ms lower. Open: the FP4 → bf16
      conversion bounds both GEMMs (0.5–0.7 TB/s against 1.25), ~7 ms a step at 32 users.
      (~~DP2×TP4 with the custom all-reduce~~: ties TP8 at 128 users, above.)
- **FP4 indexer cache:** vLLM's FP4 indexer-cache kernels crash silently on gfx90a
  (`../deepseek_v41_gfx90a`). V4-Flash uses the FP8 cache, so this run doesn't touch them.
- **To report** (laifs-container-recipes and upstream vLLM):
  1. TileLang mHC pre stage is wrong on gfx90a. Exclude all non-gfx950 ROCm, or at least
     gfx90a as gfx942 is, and attach `test_mhc_tilelang.py`. Better: offer the Triton
     big-fuse kernel (`mhc_fix.py`) as the ROCm path, which keeps the fused mHC layer; the
     96-thread/32 split is a likely wave64 cause, but re-splitting alone does not fix it.
  2. The Triton 3.6 `CanonicalizePointers` assertion.
  3. The aiter-only sparse-indexer gate (upstream main still opens it only for gfx950).
  4. The dense fp32 score in `fp8_mqa_logits_torch`.
  5. Graph pad rows are not sanitised: ROCm's `topk_hash_softplus_sqrt` returns duplicate
     ids for NaN rows, and the `triton_unfused` bitmatrix routing then corrupts *other*
     rows. Each would be a bug alone. Attach `test_topk_nan.py` and `test_moe_pad_rows.py`.
  6. The custom all-reduce works on MI250X (all GCD pairs 1 XGMI hop) if graph mode copies
     into the uncached IPC buffer as eager mode does; reading the caller's tensor directly
     returns stale data on gfx90a. Attach `test_allreduce.py`.
  7. `fp8_paged_mqa_logits_torch`'s `next_n` > 1 path (MTP without aiter) reads the indexer
     cache as interleaved per position; V4 writes it per block, so MTP indexer logits are
     wrong there. Attach `test_paged_logits.py` (section 4).
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
| `plugin/` | vLLM 0.26 general plugin: `DSV4_INDEXER_GATE`, `DSV4_TORCH_MHC`, `DSV4_TRITON_PAGED_LOGITS`, `DSV4_BF16_LINEAR` |
| `plugin/lumi_dsv4_debug/bf16_linear.py` | block-FP8 dense linears as bf16 GEMMs, FP8-rounded activations (`DSV4_BF16_LINEAR`) |
| `test_mhc_tilelang.py`, `run_mhc_test.sh` | per-kernel TileLang mHC check on one GCD |
| `plugin/lumi_dsv4_debug/mhc_fix.py` | Triton (and wave64 TileLang) mHC pre big-fuse kernel (`DSV4_MHC_FIX`); bisection and NaN-hunting switches `DSV4_MHC_NATIVE`, `_CHECK`, `_META`, `_NANTRACE`, `DSV4_NANFILL` |
| `test_mhc_fix.py`, `run_mhc_fix_test.sh` | those kernels vs the torch fallback, per-layer timing |
| `test_mhc_graph.py`, `run_mhc_graph_test.sh` | fused mHC path under CUDA graph capture/replay, padding poison |
| `plugin/lumi_dsv4_debug/paged_logits.py` | batched Triton indexer decode logits (`DSV4_TRITON_PAGED_LOGITS`) |
| `test_paged_logits.py`, `run_paged_logits_test.sh` | that kernel vs the torch fallback, graph replay, timing |
| `profile_decode.py`, `analyze_trace.py` | decode profile driver (`PROFILE=1`) and per-component GPU-time split |
| `graph_trace.py` | graph-mode traces: compute, all-reduce waiting, host-late gaps per step |
| `test_skinny_gemm_strides.py`, `run_skinny_test.sh` | ROCm skinny bf16 GEMMs with strided inputs (they are fine) |
| `bench_fp8_linear.py`, `run_fp8_linear_bench.sh`, `fp8_configs/` | dense FP8 linears: stock vs tuned vs bf16; tuned MI250X configs |
| `plugin/lumi_dsv4_debug/router_guard.py` | NaN-to-0 on MoE gate logits before top-k (`DSV4_ROUTER_GUARD`, `count` also logs how many rows) |
| `plugin/lumi_dsv4_debug/graph_diff.py` | per-layer graph-vs-eager rerun of each replayed step; first diverging site and where pad rows go non-finite (`DSV4_GRAPH_DIFF`) |
| `test_topk_nan.py`, `run_topk_nan_test.sh` | V4 router on NaN / Inf rows vs the torch reference |
| `test_moe_pad_rows.py`, `run_moe_pad_test.sh` | `triton_unfused` experts with duplicate-id pad rows; sweep over capture sizes |
| `plugin/lumi_dsv4_debug/custom_ar.py` | vLLM custom all-reduce on gfx90a, coherent via the uncached buffer or an L2 fence (`DSV4_CUSTOM_AR`) |
| `test_allreduce.py`, `run_allreduce_test.sh` | custom all-reduce vs RCCL on 8 GCDs, on changing data, graph timing |
| `plugin/lumi_dsv4_debug/score_gemm.py` | compressor score GEMMs in Triton, bf16 MFMA with fp32 output (`DSV4_SCORE_GEMM`) |
| `test_score_gemm.py`, `run_score_gemm_test.sh` | that kernel vs `torch.mm`: error vs fp64, tile sweep, per-step totals |
| `plugin/lumi_dsv4_debug/sparse_split.py` | split-K sparse attention decode for batches ≤ 16 (`DSV4_SPARSE_SPLIT`) |
| `test_sparse_split.py`, `run_sparse_split_test.sh` | split-K vs fallback decode per layer type and batch, split-count sweep |
| `plugin/lumi_dsv4_debug/sparse_decode.py` | the V4.1 port's sparse decode kernel, hooked into stock 0.26 (`DSV4_SPARSE_DECODE`) |
| `test_sparse_decode.py`, `run_sparse_decode_test.sh` | that kernel vs the fallback and split-K: accuracy, µs per call by layer type |
| `plugin/lumi_dsv4_debug/wo_a.py` | `wo_a` (`rocm_inv_rope_einsum`) as `wvSplitK` up to 3 tokens, `score_mm` above (`DSV4_WO_A`) |
| `test_wo_a.py`, `run_wo_a_test.sh` | `wo_a` sweep vs einsum / `F.linear` / vLLM, hook vs stock, split-K sum Triton vs torch |
| `plugin/lumi_dsv4_debug/dense_gemm.py` | bf16 dense linears at 2–128 tokens via `score_mm`, per-shape tiles (`DSV4_DENSE_GEMM`) |
| `plugin/lumi_dsv4_debug/mhc_fast.py` | mHC post + split-K prenorm GEMM + big fuse in Triton at decode sizes (`DSV4_MHC_FAST`) |
| `test_mhc_fast.py`, `run_mhc_fast_test.sh` | those vs the served TileLang path and the torch references, variant sweep, parts |
| `test_dense_gemm.py`, `run_dense_gemm_test.sh` | tile sweep against vLLM's ROCm GEMM dispatch for the per-rank dense shapes |
| `plugin/lumi_dsv4_debug/moe_small.py` | `pack_bitmatrix` row blocks and expert GEMM tiles at decode sizes (`DSV4_MOE_SMALL`), router GEMM via `score_mm` (`DSV4_GATE_GEMM`) |
| `test_moe_small.py`, `run_moe_small_test.sh` | `pack_bitmatrix` bit-identity and timing per block size; router GEMM sweep |
| `test_moe_matmul.py`, `run_moe_matmul_test.sh` | MXFP4 expert GEMMs: default vs forced block_n / split_k per token count |
| `plugin/lumi_dsv4_debug/moe_glue.py` | one-kernel MoE routing prologue, int32 row gathers, fused shared-expert activation (`DSV4_MOE_GLUE`) |
| `test_moe_glue.py`, `run_moe_glue_test.sh` | those vs vLLM's code: routing outputs, gathers, activation, us per MoE layer |
| `plugin/lumi_dsv4_debug/qnorm_rope.py` | attention q-norm / RoPE / KV RoPE + FP8 cache insert in one Triton kernel (`DSV4_QNORM_ROPE`) |
| `test_qnorm_rope.py`, `run_qnorm_rope_test.sh` | that kernel vs vLLM's HIP op: q, cache bytes, dequantised KV, us per call |
| `plugin/lumi_dsv4_debug/sparse_prefill.py` | sparse-attention prefill launch options from the V4.1 branch (`DSV4_SPARSE_PREFILL`) |
| `plugin/lumi_dsv4_debug/mqa_logits.py` | the indexer's prefill logits in one Triton kernel, from the V4.1 branch (`DSV4_PREFILL_LOGITS`) |
| `test_mqa_logits.py`, `run_mqa_logits_test.sh` | that kernel vs vLLM's torch fallback at V4's shapes: mask, error, top-512 overlap, time |
| `../deepseek_v41_gfx90a/prefix_cache_check.py`, `agentic_bench.py`, `prefix_evict_probe.py` | prefix caching: hit vs cold, agentic load, eviction (from the V4.1 branch; `PREFIX_CHECK` / `AGENTIC` / `PREFIX_EVICT`) |
| `test_score_prefill.py`, `run_score_prefill_test.sh` | `score_mm` tiles at prefill sizes vs `torch.mm` |
| `profile_prefill.py` | profile one long prefill (`PROFILE_PREFILL=N`), from the V4.1 branch |
| `small_kernels.py` | small kernels per step by issuing call, with Python stacks (eager traces) |
| `test_fp8_fuse.py`, `run_fp8_fuse_test.sh` | FP8 activation rounding fused into the dense GEMM: bit-identity and time (slower) |
| `test_dense_bw.py`, `run_dense_bw_test.sh` | dense bf16 GEMMs vs the memory bound: tiles, grid order, warps/stages, per-step total |
| `test_fp4_upcast.py`, `run_fp4_upcast_test.sh` | MXFP4 upcast: `tl.dot_scaled` vs bit-trick conversions vs bf16 weights in one decode-shaped GEMM |
| `test_moe_roofline.py`, `run_moe_roofline_test.sh` | expert GEMMs against HBM bandwidth and bf16 weights; block_m/n/k, split_k, warps/stages sweep |
| `../deepseek_v41_gfx90a/run_v4_serve.sh` | the serving job (switches documented at its top) |
