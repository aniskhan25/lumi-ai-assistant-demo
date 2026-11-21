"""How long a cached prefix survives other prefills (agentic_bench.py saw no hits at all).

Per round: a fresh ~9k-token prompt A (cold), then F tokens of other, unique prompts
(8 at a time, 1 output token each), then A again: its cached tokens show whether A's
blocks survived F tokens of other prefill. F = 0 first, with the filler sent at the same
time as the repeat, to separate concurrency from eviction.

    python prefix_evict_probe.py URL MODEL
"""

import json
import random
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

base, model = sys.argv[1:3]
WORDS = ("alpha beta gamma delta kernel tensor graph memory cache stream token layer "
         "expert router weight scale matrix vector compute node cluster").split()


def prompt(tokens: int) -> str:
    rng = random.Random(uuid.uuid4().int)
    return f"Doc {uuid.uuid4().hex}. " + " ".join(rng.choice(WORDS) for _ in range(tokens))


def send(p: str) -> tuple[int, int]:
    body = {"model": model, "prompt": p, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    u = json.loads(urllib.request.urlopen(req, timeout=3600).read())["usage"]
    return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def usage() -> str:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=60).read().decode()
    for line in text.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc"):
            return f"{100 * float(line.rsplit(' ', 1)[1]):.1f} %"
    return "?"


send("Warm-up.")
pool = ThreadPoolExecutor(8)
a = prompt(9000)
n, _ = send(a)
_, c = send(a)
print(f"back to back: {n} tokens, cached {c}", flush=True)
a = prompt(9000)
send(a)
futs = [pool.submit(send, prompt(9000)) for _ in range(8)]
_, c = send(a)
for f in futs:
    f.result()
print(f"repeat alongside 8 concurrent cold prompts: cached {c}", flush=True)
for filler in (20_000, 50_000, 100_000, 200_000, 400_000):
    a = prompt(9000)
    n, _ = send(a)
    t0 = time.time()
    done = sum(f.result()[0] for f in [pool.submit(send, prompt(5000))
                                       for _ in range(filler // 5000)])
    _, c = send(a)
    print(f"after {done} tokens of other prefill ({time.time() - t0:.0f} s): cached {c} of "
          f"{n}; KV usage {usage()}", flush=True)
