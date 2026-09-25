# Kimi-K2.7-Code for Aitta on LUMI-G

Per-user speed at realistic load, not peak throughput. Each combination is its own
allocation; each sweeps concurrency 1, 4, 16, 32 with streaming `vllm bench serve`
(8192 input, 512 output tokens, `--ignore-eos`) after a warm-up.

| | nodes | layout | EP | `--max-num-seqs` | `--max-num-batched-tokens` | `--max-model-len` | question |
|---|---:|---|---|---:|---:|---:|---|
| A | 2 | TP=16 PP=1 | on | 32 | 4096 | 65536 | does it fit on 2 nodes? |
| B | 3 | TP=8 PP=3 | on | 32 | 4096 | 65536 | fallback; cost of PP |
| C | 2 | TP=16 PP=1 | off | 16 | 2048 | 65536 | best achievable TPOT |
| D | 3 | TP=8 PP=3 | on | 32 | 4096 | 131072 | long-context headroom |

```bash
bash repro/kimi_k27_aitta/submit.sh        # or: ... submit.sh A
```

Pass: TPOT p99 <= ~100 ms and TTFT p99 of a few seconds at the expected concurrency.
Pick the cheapest combination that passes.

Record per job: startup_seconds (`startup.json`), the `Maximum concurrency` line
(end of `bench-mn-<id>.out`), and TTFT/TPOT p50/p99 from `serve_c*.json`.
