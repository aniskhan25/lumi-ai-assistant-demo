# DeepSeek-V4.1-Flash on LUMI-G

A vLLM plugin, packaged as a pip wheel, that serves `deepseek-ai/DeepSeek-V4.1-Flash` on
one LUMI-G node (8 × MI250X GCDs, gfx90a) in LAIF's `lumi-multitorch` container (vLLM
0.26.0), with the fastest settings measured.

## Quick start

```bash
# from the repository root, on LUMI
sbatch build.sh                                                   # once: wheel + image
MODEL=/flash/<project>/<user>/DeepSeek-V4.1-Flash sbatch serve.sh
```

`build.sh` builds the wheel (`dist/lumi_dsv41-0.1.0-py3-none-any.whl`) and the serving
image `lumi-dsv41.sif`: LAIF's image with the wheel installed into its `/opt/venv`
(`container/lumi-dsv41.def`). It takes about an hour (54 min on a `small` node). The build is unprivileged, using
LUMI's proot mode (`module load systools`).

The server is ready after ~8 minutes (`Application startup complete` in
`dsv41-serve-<jobid>.out`). It serves the OpenAI API on port 8000 of the compute node under
the name `deepseek-v4.1-flash`:

```bash
curl http://<node>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

Set `#SBATCH --account` in `build.sh` and `serve.sh` to your project. `IMAGE`, `MODEL` and
`PORT` can be overridden in the environment (`BASE` and `IMAGE` for `build.sh`); arguments
after `serve.sh` go to `vllm serve`.

## Requirements

- One LUMI-G node, all 8 GCDs.
- The LAIF base image
  `lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif` (vLLM 0.26.0), the default
  `BASE` in `build.sh`. The derived image takes 10.3 GB on disk.
- The model checkpoint (~475 GiB), ideally on `/flash`, since loading from `/scratch` is
  slower.

## Settings

| setting | value | why |
|---|---|---|
| parallelism | TP 8 + expert parallel | tensor-parallel MoE does not fit (expert shards padded to 512) |
| `--max-model-len` | 32768 | most concurrent sessions; 131072 also validated (see Tuning) |
| `--max-num-seqs` / `--max-num-batched-tokens` | 64 / 2048 | best throughput for decode and agentic load |
| `--kv-cache-dtype` | fp8 | required by V4.1's cache layout |
| `--moe-backend` | triton_unfused | the MXFP4 expert path that runs on gfx90a |
| CUDA graphs | PIECEWISE, breakable | attention and indexer run as graph breaks |
| speculative decoding | DSpark, 5 drafts | V4.1's own drafter; 1.5–1.8× decode throughput |
| prefix caching | on (vLLM default) | resent contexts skip prefill; sliding-window retention 4096 is set by the plugin |
| `VLLM_USE_V2_MODEL_RUNNER=1` | required | V4.1's Engram state needs Model Runner V2 |

## Performance (one node, natural-language prompts)

The agentic and accuracy rows were measured with this branch's image and `serve.sh`; the
others with the same kernels and settings on the development branch.

| workload | result |
|---|---|
| decode, 1k-token prompts | 64 / 471 / 730 / 963 output tok/s at 1 / 16 / 32 / 64 users |
| agentic: 32 conversations resending 8–30k-token contexts | 171–174 output tok/s, 87 % of prompt tokens from the cache, median time to first token 3.1–7.3 s |
| resent 17k-token context | first token in ~0.2 s, against 4.5 s uncached |
| cold prefill, one request | time to first token 1.2 / 3.8 / 7.0 s at 4k / 16k / 30k tokens |
| accuracy | GSM8K 98–99 % (200 questions) |

## Tuning

- **Long contexts:** `sbatch serve.sh --max-model-len 131072` (the model supports up to
  1M; 128k is the validated range). Tested with the image:
  - retrieval: needles found 9/9 at ~31k / 62k / 114k tokens, at 10 / 50 / 90 % depth;
  - accuracy: GSM8K 98.5 %, no out-of-memory errors;
  - cold time to first token: ~7 / 15 / 30 s at 31k / 62k / 114k;
  - KV cache: 3.2M tokens, 24 full-length requests.

  Decode is unchanged at 1 user and 2.6 / 3.3 % slower at 32 / 64 users (682 / 917 against
  700 / 948 tok/s). The decode indexer's score buffer is sized by `--max-model-len`, not
  by the actual context.

- **Interactive agents:** `sbatch serve.sh --max-num-batched-tokens 8192` roughly halves
  time to first token under load. Tokens stream about 10–20 % slower.
- **Throughput-bound serving:** `--max-num-seqs 128` gives 970 tok/s at 128 users.
- **Prefix caching:** it helps only while the conversations' live contexts fit in the KV
  cache (2.24M tokens at `--gpu-memory-utilization 0.92`, enough for 64 conversations of
  up to 30k tokens). Beyond that, hits drop to zero. Turn it off with
  `--no-enable-prefix-caching`.
- **Expert placement:** experts are spread over the 8 GCDs by measured popularity
  (`lumi_dsv41/expert_placement.json`, +4 % on agentic load). The placement was
  measured on GSM8K-style text. `DSV41_EXPERT_PLACEMENT=none` restores vLLM's default
  placement; `DSV41_EXPERT_PLACEMENT=<file>` uses another file in the same format.

## What the plugin does

vLLM 0.26 has DeepSeek-V4 but not V4.1, and its V4 ROCm path needs aiter or TileLang
kernels that do not run correctly on gfx90a. The plugin:

- **Loads upstream's V4.1 model code** (`lumi_dsv41/vendor`, the AMD path, language model only)
  into 0.26. Small hooks adapt it to 0.26's KV-cache, scheduler and config interfaces.
- **Adds the missing gfx90a kernels** (Triton):
  - sparse-attention decode;
  - indexer prefill and decode logits;
  - mHC (hyper-connection) pre and post;
  - the query RoPE.
  It also sets faster launch settings for the upstream sparse-attention prefill kernel.
- **Speeds up the dense layers and communication:**
  - runs the block-FP8 dense layers as bf16 GEMMs (gfx90a has no FP8 units);
  - enables vLLM's custom all-reduce on gfx90a, made coherent across GCDs;
  - tunes the MoE routing blocks and expert GEMM tiles.
- **Fixes correctness issues:**
  - guards the MoE router against NaN rows in graph padding;
  - fixes two upstream gfx90a bugs: decode inverse RoPE, and the tiled indexer-cache read.
- **Enables the rest of the configuration:** prefix caching without replay, the DSpark
  drafter, and the balanced expert placement.

## Layout

```
pyproject.toml                the package: lumi-dsv41, vLLM plugin entry point, no dependencies
lumi_dsv41.pth                installs the module loader at interpreter start (site-packages)
lumi_dsv41/                   the plugin
  vendor/                     upstream DeepSeek-V4.1 modules, loaded in place of 0.26's
container/lumi-dsv41.def      serving image: LAIF base + the wheel
build.sh                      Slurm script: wheel + image
serve.sh                      Slurm script: one node, best settings
```

## Installing the wheel elsewhere

`pip install --no-deps lumi_dsv41-0.1.0-py3-none-any.whl` into any Python environment
that has LAIF's vLLM 0.26.0. `--no-deps` keeps pip away from the container's vLLM, torch
and Triton; the package declares no dependencies anyway.

It must go into a real site-packages directory. `lumi_dsv41.pth` loads the plugin's module
loader when Python starts, before vLLM is imported, and Python reads `.pth` files only from
site-packages, not from `PYTHONPATH` or `--target` directories. LAIF's `/opt/venv` is
read-only inside the image, which is why `build.sh` bakes the wheel into a derived image.

Then serve with the arguments in `serve.sh`, and set `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` and `AMDGCN_USE_BUFFER_OPS=0`.

## Limitations

- One node only (TP 8 with expert parallelism); validated up to 128k tokens of context
  (prompts to ~114k).
- Language model only: no vision input.
- gfx90a (MI250X) with this container only.
- DSpark is the only speculative method.
- If vLLM fails at start-up with a shared-memory broadcast error, check that `/dev/shm`
  is private. `serve.sh` binds a per-job directory for this.

The development history, measurements and tests are on branch `exp/deepseek-v41-gfx90a`.
