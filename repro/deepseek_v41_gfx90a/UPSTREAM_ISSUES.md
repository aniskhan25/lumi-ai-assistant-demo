# Issue drafts from the DeepSeek-V4.1-Flash port to LUMI-G (MI250X, gfx90a)

Drafts only, not filed. Part A is for upstream vLLM (code at `vllm-project/vllm@65711a1`,
vendored in `vendor/`); it affects any ROCm GPU other than gfx942/gfx950, or any setup
without aiter. Part B is for whoever runs upstream's V4.1 code on vLLM 0.26 (the LAIF
container, laifs-container-recipes): mismatches between 0.26's engine and upstream's
model code, each worked around in `lumi_dsv41/`. Job numbers refer to LUMI runs logged
in `README.md`.

## A1. ROCm sparse decode: the fallback kernel skips the inverse RoPE but reports it done

**Where.** `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, `rocm_sparse_attn_decode`,
called from `vllm/models/deepseek_v41/amd/rocm.py`.

**What.** Given `inv_rope_positions`, `rocm_sparse_attn_decode` returns `output.shape[0]`,
telling the caller every decode row is already inverse-RoPE'd. Only the gfx942/gfx950
split-K reduce applies it (`FUSE_INV_ROPE`); on other GPUs
`_rocm_sparse_attn_decode_ragged_triton` takes the one-pass fallback and returns before
any inverse RoPE. The caller trusts the count and skips its own `rocm_inverse_rope_rows_`
for those rows, so every decode token reaches `wo_a` with a rotated RoPE tail.

**Symptom.** Prefill is right, decode drifts: fluent but wrong text, decode and prefill
log-probabilities of the same tokens differ by up to 23 nats, GSM8K 3 % with 36/200
garbage answers (job 23641719). After the fix: GSM8K 98–99 %.

**Fix.** Return 0 (or the number of rows actually rotated) when the fallback runs, or
apply the inverse RoPE in the fallback kernel. Our workaround withholds the positions off
gfx942/950 so the caller rotates (`hooks.py:_decode_inv_rope`).

## A2. ROCm indexer: decode logits read the 16×16-tiled K cache row-major

**Where.** `vllm/models/deepseek_v41/common/ops/indexer_k_store.py` (writer) and
`fp8_paged_mqa_logits_torch` in `rocm_aiter_mla_sparse.py` (reader without aiter).

**What.** On ROCm the indexer K writer stores each block's FP8 values in a 16×16-tiled
("SHUFFLE") layout: byte `b` of position `p` sits at `[p // 16, b // 16, p % 16, b % 16]`
within the block. Prefill's gather and aiter's Preshuffle decode kernel read it tile-aware.
Without aiter, decode falls back to `fp8_paged_mqa_logits_torch`, which reads the block
row-major, so every key is a permutation of bytes from 16 positions.

**Symptom.** Short prompts are fine (the top-k covers every compressed position anyway).
From about 3k tokens, once a context has more compressed positions than `index_topk`
(512), decode picks the wrong top-512 and diverges from prefill from the 5th generated
token on; a needle at 16.8k tokens is answered `482` instead of `48213` (job 23643490).
After the fix, decode matches prefill to 0.02–0.05 nats up to 20k tokens (job 23644138).

**Fix.** Un-tile in the fallback (or make the writer's layout depend on whether a
tile-aware reader exists). Our workaround un-tiles each gathered page
(`hooks.py:_paged_logits_unshuffle`); our batched Triton kernel reads the tiles directly
(`lumi_dsv41/paged_logits.py`, `SHUFFLE` constexpr).

## A3. Performance: sparse MLA decode off gfx942/950 runs at ~50 GB/s

**Where.** `_sparse_attn_decode_ragged_kernel` (fallback) and
`_sparse_attn_decode_partial_kernel` (split-K) in `rocm_aiter_mla_sparse.py`.

**What.** Both paths share an inner loop that walks 16 KV entries per iteration, gathers
the UE8M0 scale byte once per element (16 × 512 loads per iteration), and computes the 448
FP8 NoPE dims (padded to 512) and the 64 bf16 RoPE dims as two dot pairs. The fallback
also runs one workgroup per query and head block. On MI250X, a top-512 layer at 64 queries
moves ~24 MB in ~450 µs. In V4.1 (38 top-k layers) this was 28 % of a 32-user decode step.

**Offer.** `lumi_dsv41/sparse_decode.py`: split-K partial + reduce, 8 scale bytes per
token, one [16, 512] bf16 K tile (448 dequantised + 64 RoPE) per step, splits by batch and
layer type; same results within bf16 rounding (`test_sparse_decode.py`). Per top-k layer
call on one GCD: 410 → 30 µs at 1 query, 438 → 108 at 32, 447 → 189 at 64, 891 → 339 at
128 (the split-K path: 62 / 281 / 584 / 1266). Serving: +6–9 % at 1–64 users. Not tuned
beyond gfx90a.

## B. vLLM 0.26 vs upstream V4.1 (LAIF container)

Each item: what 0.26 does, what upstream V4.1 expects, and our workaround.

| # | 0.26 behaviour | upstream expectation | effect | workaround |
|---|---|---|---|---|
| B1 | Model Runner V2 PIECEWISE graphs pad the token count to the graph size but not the request count | padded requests | decode indexer stores 15 unpacked top-k rows into a 16-token slice: shape error, engine dead (job 23666313) | `_indexer_graph_pad`: store real rows, mark pad tokens -1 |
| B2 | a lone small prefill (e.g. the last 78-token chunk of a long prompt) is padded to `max_num_seqs` zero-length requests | — | `_forward_prefill` counts 63 empty prefills, plans an all-empty chunk, `view(22, 0, -1)` fails (jobs 23836592/3, async + 2048-token chunks) | `_empty_prefill_chunks`: skip chunks with no tokens |
| B3 | breakable CUDA graphs are auto-enabled only for built-in model names | — | a plugin model runs eager under `COMPILE=1`, with only a log line ("torch.compile is turned on, but the model ... does not support it") | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| B4 | `config.utils.replace` rebuilds from `__dict__` and raises on non-field keys; `AttentionConfig.indexer_kv_dtype` is a Literal without `"auto"` | `indexer_kv_dtype="auto"`, `sparse_mla_prefill_workspace_factor` | the DSpark loader's `replace(attention_config)` fails (jobs 23861137, 23861245) | `_config_replace` carries them over |
| B5 | the runner slices `get_mtp_target_hidden_states()` unconditionally | returns None for DSpark (it drafts from aux hidden states) | `'NoneType' object is not subscriptable` in the profiling run (job 23861746) | target returns the step's hidden states then |
| B6 | `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` takes 9 arguments and always applies V4's per-head query norm | 11 arguments (`apply_q_norm`, `kv_mxfp8`, `apply_q_rope`, `is_q_interleaved`); V4.1 has no query norm | wrong query (attention) and a TypeError (DSpark context-KV insert, job 23861880) | `ops.py` wrapper (KV side on 0.26's kernel, query in Triton) |
| B7 | `qwen3_dspark` has no `DSparkConfidenceHead`; the DSpark speculator does no adaptive verification | the V4.1 drafter imports and builds it | ImportError | stand-in class; head disabled (`enable_confidence_head = False`) |
| B8 | DSpark accepts only `num_speculative_tokens >= dspark_block_size` (5); with 7 the DeepSeek-V4 KV page packer (`_get_kv_cache_groups_uniform_groups`) asserts (job 23868852) | — | only 5 drafts usable | none (5 is the trained block size) |
| B9 | `RocmPlatform.use_custom_allreduce` is False on gfx90a; the custom kernel reads peers' inputs over XGMI from cached memory | — | RCCL only; forcing the custom kernel gives stale data (GSM8K 0 % on V4) | V4 plugin `custom_ar.py`: copy through the uncached IPC buffer (+6 %) |
