"""Bring-up smoke requests: every request must finish with tokens (content is irrelevant
with random weights). Short, ~3k and ~12k-token prompts (compressed/sparse paths, which
start past the 128-token window), then a concurrent mixed batch, then decode-heavy."""

import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
FILLER = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "


def complete(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return {"prompt_tokens": out["usage"]["prompt_tokens"],
            "completion_tokens": out["usage"]["completion_tokens"],
            "text": out["choices"][0]["text"][:40]}


def run(name, prompt, max_tokens):
    t = time.time()
    try:
        r = complete(prompt, max_tokens)
        ok = r["completion_tokens"] == max_tokens
        print(f"[{name}] ok={ok} {r} {time.time() - t:.1f}s", flush=True)
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAILED {type(e).__name__}: {e}", flush=True)
        return False


results = [run("short", "Hello, my name is", 16),
           run("3k", FILLER * 200, 16),
           run("12k", FILLER * 800, 16)]
with cf.ThreadPoolExecutor(8) as pool:
    jobs = [pool.submit(run, f"mix{i}", FILLER * (5 + 300 * (i % 3)), 32) for i in range(8)]
    results += [j.result() for j in jobs]
results.append(run("decode", "Count:", 256))
print(f"SMOKE {'PASS' if all(results) else 'FAIL'} {sum(results)}/{len(results)}")
sys.exit(0 if all(results) else 1)
