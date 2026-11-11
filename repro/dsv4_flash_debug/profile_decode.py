"""Drive a running vLLM server into steady decode at a given concurrency, then profile.

Starts `concurrency` streaming completions (random-ish ~1k-token prompts, ignore_eos), waits
until every one is decoding, calls /start_profile, waits, calls /stop_profile, and lets the
requests finish. The server must run with --profiler-config profiler=torch.

    python profile_decode.py http://127.0.0.1:PORT MODEL CONCURRENCY [SECONDS]
"""

import json
import random
import sys
import threading
import time
import urllib.request

base, model, concurrency = sys.argv[1], sys.argv[2], int(sys.argv[3])
window = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
WORDS = ("sopimus laki työ data model kernel tensor graph memory cache stream token "
         "layer expert router weight scale matrix vector compute node cluster").split()

started = [0] * concurrency
lock = threading.Lock()


def run(i: int) -> None:
    rng = random.Random(i)
    prompt = " ".join(rng.choice(WORDS) for _ in range(900))
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 160, "stream": True,
                       "ignore_eos": True, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                with lock:
                    started[i] += 1


def post(path: str) -> None:
    urllib.request.urlopen(urllib.request.Request(f"{base}{path}", b"", method="POST"),
                           timeout=3600).read()


threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(concurrency)]
for t in threads:
    t.start()
t0 = time.time()
while min(started) < 4:  # every request past prefill and decoding
    if time.time() - t0 > 600 or not any(t.is_alive() for t in threads):
        sys.exit(f"requests did not reach decode: {started}")
    time.sleep(0.2)
before = sum(started)
post("/start_profile")
time.sleep(window)
post("/stop_profile")
print(f"profiled c={concurrency}: {sum(started) - before} tokens streamed during the window")
for t in threads:
    t.join()
