"""Preemption of requests that started from a prefix-cache hit.

Requests share a ~PREFIX-token few-shot prefix (GSM8K questions with worked answers), so
each starts from a hit, and ask one more question, greedy, OUT tokens with EOS ignored: the
answer comes early, then the model keeps writing more questions and answers, so every
request's KV grows. Run on a small KV pool (--gpu-memory-utilization), the growth forces
vLLM to preempt running requests mid-generation and resume them later (recomputing, with
another prefix lookup). Two phases with the same requests:

  reference: CONC_REF at a time, few enough that nothing is preempted;
  pressure:  all N at once.

Greedy output is not repeatable here, so the phases are compared by statistics: answer
accuracy, the mean logprob of the first and last 1000 generated tokens (state corrupted on
resume would show as a drop in the tail), a repetition check, and preemptions from
/metrics.

    python preempt_check.py URL MODEL GSM8K_PARQUET [N] [PREFIX] [OUT] [CONC_REF]
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

base, model, parquet = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 64
PREFIX = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
OUT = int(sys.argv[6]) if len(sys.argv) > 6 else 6000
CONC_REF = int(sys.argv[7]) if len(sys.argv) > 7 else 16

df = pd.read_parquet(parquet)
shots, n = [], 0
for q, a in zip(df.question[:500], df.answer[:500]):
    shots.append(f"Question: {q}\nAnswer: {a}\n\n")
    n += len(shots[-1])
    if n > PREFIX * 3.6:
        break
prefix = "".join(shots)
questions = list(zip(df.question[600:600 + N], df.answer[600:600 + N]))


def gold(a: str) -> str:
    return a.split("####")[-1].strip().replace(",", "")


def first_answer(text: str) -> str | None:
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def preemptions() -> float:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=60).read().decode()
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
               if line.startswith("vllm:num_preemptions_total"))


def run(k: int) -> dict:
    q, a = questions[k]
    body = {"model": model, "prompt": prefix + f"Question: {q}\nAnswer:", "max_tokens": OUT,
            "temperature": 0.0, "ignore_eos": True, "logprobs": 1}
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    choice, usage = out["choices"][0], out["usage"]
    lp = [x for x in choice["logprobs"]["token_logprobs"] if x is not None]
    toks = choice["logprobs"]["tokens"]
    tail = toks[-1000:]
    return {"right": first_answer(choice["text"]) == gold(a),
            "head": sum(lp[:1000]) / max(len(lp[:1000]), 1),
            "tail": sum(lp[-1000:]) / max(len(lp[-1000:]), 1),
            # a corrupted state often loops: share of the tail's most common 8-gram
            "loop": max((sum(1 for i in range(len(tail) - 8)
                             if tail[i:i + 8] == tail[j:j + 8]) for j in range(0, len(tail) - 8, 97)),
                        default=0) / max(len(tail) - 8, 1),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "out": usage["completion_tokens"]}


def phase(name: str, conc: int) -> list[dict]:
    p0, t0 = preemptions(), time.time()
    with ThreadPoolExecutor(conc) as pool:
        rs = list(pool.map(run, range(N)))
    p1 = preemptions()
    mean = lambda key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
    print(f"{name:9} ({conc:2d} at a time, {time.time() - t0:5.0f} s): preemptions {p1 - p0:.0f}; "
          f"right {sum(r['right'] for r in rs)}/{N}; mean logprob first 1000 {mean('head'):.3f}, "
          f"last 1000 {mean('tail'):.3f} (worst {min(r['tail'] for r in rs):.3f}); "
          f"8-gram loop share mean {mean('loop'):.3f} max {max(r['loop'] for r in rs):.3f}; "
          f"cached {mean('cached'):.0f} of the prompt on average", flush=True)
    return rs


run(0)  # warm the shared prefix
ref = phase("reference", CONC_REF)
hot = phase("pressure", N)
gap = sum(r["tail"] for r in ref) / N - sum(r["tail"] for r in hot) / N
ok = (sum(r["right"] for r in hot) >= sum(r["right"] for r in ref) - max(2, N // 16)
      and gap < 0.1 and max(r["loop"] for r in hot) <= max(0.2, 2 * max(r["loop"] for r in ref)))
print("PREEMPT_CHECK", "OK" if ok else "FAILED",
      f"(tail logprob gap {gap:.3f}; preemptions must be > 0 in the pressure phase to count)")
sys.exit(0 if ok else 1)
