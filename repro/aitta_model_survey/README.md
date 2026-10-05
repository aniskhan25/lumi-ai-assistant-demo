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
