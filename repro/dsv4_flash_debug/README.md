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

## Open

- **Speed:** only eager has been tried. The indexer-gate plugin calls the op's Python
  implementation, because 0.26 registers the `torch.ops` wrapper only with aiter.
  torch.compile would need it registered. GSM8K took 742 s, against 233 s for compiled
  Qwen3.8-Flash.
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
| `plugin/` | vLLM 0.26 general plugin: `DSV4_INDEXER_GATE`, `DSV4_TORCH_MHC` |
| `test_mhc_tilelang.py`, `run_mhc_test.sh` | per-kernel TileLang mHC check on one GCD |
| `../deepseek_v41_gfx90a/run_v4_serve.sh` | the serving job (switches documented at its top) |
