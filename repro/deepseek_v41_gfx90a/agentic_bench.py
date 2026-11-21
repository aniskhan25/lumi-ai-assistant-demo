"""Agentic load: concurrent conversations that resend a growing context every turn.

Each conversation is a shared preamble, its own session line and a ~START-token document
(GSM8K test questions with worked answers). A turn sends the whole context plus a new
question, greedy, up to OUT tokens; then the question, the answer and a ~GROW-token "tool
output" (more worked examples) are appended. At END tokens the conversation starts over
with a new document. CONVS conversations run at once for DURATION seconds; turns that
finish in the window after WARMUP count.

Reports output throughput, time to first token and time per output token by context
length, the share of prompt tokens served from the prefix cache, GSM8K accuracy of the
answers, preemptions and DSpark acceptance (from /metrics). Run once with
PREFIX_CACHING=1 and once without, in separate allocations.

    python agentic_bench.py URL MODEL GSM8K_PARQUET OUT_JSONL [CONVS ...]
    env: AGENT_DURATION (1200 s), AGENT_WARMUP (300 s), AGENT_START (8000),
         AGENT_END (30000), AGENT_GROW (1500), AGENT_OUT (384)
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
import uuid

import pandas as pd

base, model, parquet, out_path = sys.argv[1:5]
CONVS = [int(c) for c in sys.argv[5:]] or [32, 64]
DURATION = float(os.environ.get("AGENT_DURATION", 1200))
WARMUP = float(os.environ.get("AGENT_WARMUP", 300))
START = int(os.environ.get("AGENT_START", 8000))
END = int(os.environ.get("AGENT_END", 30000))
GROW = int(os.environ.get("AGENT_GROW", 1500))
OUT = int(os.environ.get("AGENT_OUT", 384))
CHARS_PER_TOKEN = 3.6  # GSM8K text, roughly; real counts come from usage

df = pd.read_parquet(parquet)
SHOTS = [f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(df.question[:500], df.answer[:500])]
QUESTIONS = list(zip(df.question[500:], df.answer[500:]))
PREAMBLE = ("You are a careful assistant working through a long session. Earlier questions, "
            "worked answers and tool outputs follow. Answer the last question step by step "
            "and end with '#### <number>'.\n\n")


def text_of(start: int, tokens: int) -> str:
    out, n, i = [], 0, start
    while n < tokens * CHARS_PER_TOKEN:
        out.append(SHOTS[i % len(SHOTS)])
        n += len(out[-1])
        i += 1
    return "".join(out)


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def predicted(text: str) -> str | None:
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def metrics() -> dict[str, float]:
    names = ("vllm:num_preemptions_total", "vllm:spec_decode_num_accepted_tokens_total",
             "vllm:spec_decode_num_drafts_total", "vllm:prefix_cache_hits_total",
             "vllm:prefix_cache_queries_total")
    got = dict.fromkeys(names, 0.0)
    for line in urllib.request.urlopen(f"{base}/metrics", timeout=60).read().decode().splitlines():
        for name in names:
            if line.startswith(name):
                got[name] += float(line.rsplit(" ", 1)[1])
    return got


def turn(prompt: str) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": OUT, "temperature": 0.0,
            "stop": ["\n\nQuestion:"], "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                if ttft is None:
                    ttft = time.time() - t0
                text.append(choice.get("text") or "")
    return {"text": "".join(text), "ttft": ttft, "e2e": time.time() - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "out_tokens": usage.get("completion_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}


def conversation(cid: int, n_convs: int, t_start: float, records: list, lock) -> None:
    restart, qi = 0, 0
    while time.time() - t_start < DURATION:
        session = uuid.uuid4().hex
        context = (PREAMBLE + f"Session {session}.\n\n"
                   + text_of((cid * 13 + restart * 101) % len(SHOTS), START))
        tokens = 0
        while tokens < END and time.time() - t_start < DURATION:
            q, a = QUESTIONS[(cid * 97 + qi) % len(QUESTIONS)]
            qi += 1
            prompt = context + f"Question: {q}\nAnswer:"
            try:
                r = turn(prompt)
            except Exception as e:  # noqa: BLE001 - count, keep the load going
                r = {"error": repr(e), "prompt_tokens": 0, "out_tokens": 0}
            r.update(conv=cid, convs=n_convs, restart=restart, done=time.time() - t_start,
                     right=predicted(r.get("text", "")) == gold(a))
            with lock:
                records.append(r)
            if "error" in r:
                return
            tokens = r["prompt_tokens"] + r["out_tokens"]
            context = (prompt + r.pop("text") + "\n\nTool output:\n"
                       + text_of((cid * 31 + qi * 7) % len(SHOTS), GROW))
        restart += 1


def bucket(n: int) -> str:
    lo = n // 8000 * 8
    return f"{lo:2d}-{lo + 8}k"


def report(n_convs: int, records: list, m0: dict, m1: dict) -> None:
    window = [r for r in records if WARMUP <= r["done"] <= DURATION and "error" not in r]
    errors = [r for r in records if "error" in r]
    span = DURATION - WARMUP
    out = sum(r["out_tokens"] for r in window)
    prompt = sum(r["prompt_tokens"] for r in window)
    cached = sum(r["cached"] for r in window)
    right = sum(r["right"] for r in records if "error" not in r)
    n_all = sum("error" not in r for r in records)
    d = {k: m1[k] - m0[k] for k in m0}
    drafts = d["vllm:spec_decode_num_drafts_total"]
    acc = (1 + d["vllm:spec_decode_num_accepted_tokens_total"] / drafts) if drafts else 0
    hit = (d["vllm:prefix_cache_hits_total"] / d["vllm:prefix_cache_queries_total"]
           if d["vllm:prefix_cache_queries_total"] else 0)
    print(f"--- {n_convs} conversations, {span:.0f} s window: {len(window)} turns, "
          f"{out / span:.0f} output tok/s, {prompt / span:.0f} prompt tok/s of which "
          f"{100 * cached / max(prompt, 1):.1f} % cached (engine hit rate {100 * hit:.1f} %); "
          f"answers right {right}/{n_all}; preemptions "
          f"{d['vllm:num_preemptions_total']:.0f}; DSpark acceptance {acc:.2f}; "
          f"errors {len(errors)}", flush=True)
    for b in sorted({bucket(r["prompt_tokens"]) for r in window}):
        rs = [r for r in window if bucket(r["prompt_tokens"]) == b]
        ttft = sorted(r["ttft"] for r in rs)
        tpot = sorted((r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1)
                      for r in rs if r["out_tokens"] > 1)
        print(f"  context {b}: {len(rs):4d} turns, TTFT median {ttft[len(ttft) // 2]:.2f} s "
              f"p90 {ttft[int(len(ttft) * 0.9)]:.2f} s, TPOT median "
              f"{1e3 * tpot[len(tpot) // 2]:.1f} ms p90 {1e3 * tpot[int(len(tpot) * 0.9)]:.1f}"
              f" ms, cached {100 * sum(r['cached'] for r in rs) / sum(r['prompt_tokens'] for r in rs):.0f} %, "
              f"right {sum(r['right'] for r in rs)}/{len(rs)}", flush=True)
    if errors:
        print("  first error:", errors[0]["error"][:300])


turn("Warm-up.")
with open(out_path, "w") as f:
    for n_convs in CONVS:
        records, lock = [], threading.Lock()
        m0 = metrics()
        t_start = time.time()
        threads = [threading.Thread(target=conversation, args=(c, n_convs, t_start, records, lock))
                   for c in range(n_convs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        m1 = metrics()
        report(n_convs, records, m0, m1)
        for r in records:
            f.write(json.dumps(r) + "\n")
