"""Profile one long prefill: /start_profile, a single request with a prompt of about N
tokens and 1 output token, /stop_profile. The server must run with --profiler-config
profiler=torch. Prints the prompt's token count and the time to the first token.

With SUFFIX (an agentic turn): the profiled request is a cached ~N-token prefix plus a new
~SUFFIX-token suffix, so only the suffix is computed, attending over the whole prefix. The
server needs prefix caching (and --enable-prompt-tokens-details to print the hit).

    python profile_prefill.py http://127.0.0.1:PORT MODEL APPROX_TOKENS [SUFFIX_TOKENS]
"""

import json
import random
import sys
import time
import urllib.request

base, model, n_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
suffix_tokens = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0
WORDS = ("sopimus laki työ data model kernel tensor graph memory cache stream token "
         "layer expert router weight scale matrix vector compute node cluster").split()


def post(path: str, body: bytes = b"") -> bytes:
    req = urllib.request.Request(f"{base}{path}", body, {"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=3600).read()


def text(words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words))


def send(prompt: str) -> tuple[int, int, float]:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    t0 = time.time()
    usage = json.loads(post("/v1/completions", body))["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return usage["prompt_tokens"], cached, time.time() - t0


def complete(words: int) -> tuple[int, float]:
    tok, _, dt = send(text(words, words))
    return tok, dt


# Calibrate words per token on a short prompt (outside the profile), then warm the shape.
tok, _ = complete(2000)
words = int(2000 * n_tokens / tok)
if not suffix_tokens:
    complete(words)
    post("/start_profile")
    tok, dt = complete(words + 1)  # a different prompt
    post("/stop_profile")
    print(f"profiled prefill: {tok} prompt tokens, {dt * 1e3:.0f} ms to the first token "
          f"({tok / dt:.0f} tokens/s)")
    sys.exit(0)

suffix_words = int(2000 * suffix_tokens / tok)
# Same shapes outside the profile: a cold suffix-sized prompt, then a cached prefix with a
# new suffix. The suffix starts with a newline so the prefix tokenizes the same.
_, _, cold = send(text(suffix_words, 7))
prefix = text(words, 1)
send(prefix)
send(prefix + "\n" + text(suffix_words, 2))
post("/start_profile")
tok, cached, dt = send(prefix + "\n" + text(suffix_words, 3))
post("/stop_profile")
print(f"profiled agentic turn: {tok} prompt tokens, {cached} cached, {tok - cached} computed "
      f"over the cached context, {dt * 1e3:.0f} ms to the first token "
      f"({(tok - cached) / dt:.0f} computed tokens/s; the same length cold at position 0: "
      f"{cold * 1e3:.0f} ms)")
