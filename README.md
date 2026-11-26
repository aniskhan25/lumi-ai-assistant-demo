# DeepSeek-V4.1-Flash on LUMI-G

A vLLM plugin (pip wheel) that serves `deepseek-ai/DeepSeek-V4.1-Flash` on one LUMI-G node
(8 × MI250X GCDs) in LAIF's `lumi-multitorch` container (vLLM 0.26.0), with the fastest
settings measured.

## Serve it

1. **Clone** (on LUMI):
   ```bash
   git clone --branch prod/deepseek-v41-gfx90a --single-branch \
       https://github.com/aniskhan25/lumi-ai-assistant-demo.git dsv41 && cd dsv41
   ```
2. **Get the model** (~475 GiB) onto `/flash`, which loads faster than `/scratch`. For
   example:
   ```bash
   BASE=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif
   singularity exec $BASE hf download deepseek-ai/DeepSeek-V4.1-Flash \
       --local-dir /flash/<project>/$USER/DeepSeek-V4.1-Flash
   ```
3. **Set your project** in `#SBATCH --account` of `build.sh` and `serve.sh`.
4. **Build the image** once (54 min, 10.3 GB): `sbatch build.sh` → `lumi-dsv41.sif`.
5. **Start the server:**
   ```bash
   MODEL=/flash/<project>/$USER/DeepSeek-V4.1-Flash sbatch serve.sh
   ```
   It is ready after ~9 minutes, when `dsv41-serve-<jobid>.out` shows
   `Application startup complete`. `squeue --me` shows the node.
6. **Use it** from your laptop through a tunnel:
   ```bash
   ssh -N -L 8000:<node>:8000 <user>@lumi.csc.fi
   curl http://localhost:8000/v1/models
   ```
   Any OpenAI client then works with `base_url="http://localhost:8000/v1"`, any `api_key`
   string, and `model="deepseek-v4.1-flash"`.
7. **Stop it:** `scancel <jobid>`. The node is billed while the job runs.

`IMAGE`, `MODEL` and `PORT` can be set in the environment. Arguments after `serve.sh` go to
`vllm serve`.

## Settings

| setting | value | why |
|---|---|---|
| parallelism | TP 8 + expert parallel | tensor-parallel MoE does not fit (expert shards padded to 512) |
| `--max-model-len` | 131072 | room for thinking, agentic and long-document work; validated to ~114k-token prompts |
| `--max-num-seqs` / `--max-num-batched-tokens` | 64 / 2048 | best decode and agentic throughput |
| `--kv-cache-dtype` | fp8 | required by V4.1's cache layout |
| `--moe-backend` | triton_unfused | the MXFP4 expert path that runs on gfx90a |
| CUDA graphs | PIECEWISE, breakable | attention and indexer run as graph breaks |
| speculative decoding | DSpark, 5 drafts | V4.1's own drafter: 1.5–1.8× decode throughput |
| prefix caching | on | sliding-window retention 4096, set by the plugin |

## Performance (one node, natural-language prompts)

| workload | result |
|---|---|
| decode, 1k-token prompts | 62 / 682 / 917 output tok/s at 1 / 32 / 64 users |
| agentic: 32 conversations resending 8–30k-token contexts¹ | 171–174 output tok/s, 87 % of prompt tokens from the cache, median time to first token 3.1–7.3 s |
| resent 17k-token context¹ | first token in ~0.2 s, against 4.5 s uncached |
| cold prefill, one request | time to first token 1.2 / 3.8 / 7 / 15 / 30 s at 4k / 16k / 31k / 62k / 114k tokens |
| long-context retrieval | needles found 9/9 at ~31k / 62k / 114k tokens, 10 / 50 / 90 % depth |
| accuracy | GSM8K 98.5–99 % (200 questions), no out-of-memory errors |
| KV cache | 3.2M tokens, room for 24 requests at the full 131k |

¹ At `--max-model-len 32768`. The resent-context row and the prefill times below 31k come
from the development branch, with the same kernels.

## Tuning

- **Many short sessions:** `--max-model-len 32768`. Decode is 2.6 / 3.3 % faster at
  32 / 64 users (700 / 948 tok/s), because the decode indexer's score buffer is sized by
  the limit, not the actual context.
- **Interactive agents:** `--max-num-batched-tokens 8192` roughly halves time to first
  token under load; tokens stream 10–20 % slower.
- **Throughput:** `--max-num-seqs 128` gives 970 tok/s at 128 users.
- **Prefix caching** helps only while live contexts fit in the KV cache. Each request also
  holds a fixed sliding-window share, so 64 concurrent conversations fit about 36k tokens
  each.
- **Expert placement:** experts are spread over the GCDs by measured popularity (+4 % on
  agentic load, measured on GSM8K-style text). `DSV41_EXPERT_PLACEMENT=none` restores
  vLLM's default placement.

The prompt plus output must fit in `--max-model-len`, and with thinking on the reasoning
counts as output. The model supports up to 1M tokens (YaRN ×16 over a 64k training window);
128k is the validated range.

## What the plugin does

vLLM 0.26 has DeepSeek-V4 but not V4.1, and its ROCm path needs aiter or TileLang kernels
that do not run correctly on gfx90a. The plugin:

- **loads upstream's V4.1 model code** (`lumi_dsv41/vendor`) into 0.26;
- **adds the missing gfx90a kernels** in Triton: sparse-attention decode, indexer logits,
  mHC and the query RoPE;
- **speeds up the dense layers and communication:** block-FP8 dense layers as bf16 GEMMs
  (no FP8 units on gfx90a), a coherent custom all-reduce, tuned MoE tiles;
- **fixes two upstream gfx90a bugs** (decode inverse RoPE, the tiled indexer-cache read)
  and guards the MoE router against NaN padding rows;
- **enables** prefix caching without replay, the DSpark drafter and the balanced expert
  placement.

The wheel can also be installed with `pip install --no-deps` into any environment with
LAIF's vLLM 0.26.0. It must go into a real site-packages directory, where
`lumi_dsv41.pth` starts the plugin with Python. LAIF's image is read-only, hence the
derived image.

## Limitations

- One node; validated to 128k context.
- Language model only: no vision input.
- gfx90a (MI250X) with this container only.
- DSpark is the only speculative method.

Development history, measurements and tests: branch `exp/deepseek-v41-gfx90a`.
