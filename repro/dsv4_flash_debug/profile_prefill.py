"""Profile one long prefill: /start_profile, a single request with a prompt of about N
tokens and 1 output token, /stop_profile. The server must run with --profiler-config
profiler=torch. Prints the prompt's token count and the time to the first token.

    python profile_prefill.py http://127.0.0.1:PORT MODEL APPROX_TOKENS
"""

import json
import random
import sys
import time
import urllib.request

base, model, n_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
WORDS = ("sopimus laki työ data model kernel tensor graph memory cache stream token "
         "layer expert router weight scale matrix vector compute node cluster").split()


def post(path: str, body: bytes = b"") -> bytes:
    req = urllib.request.Request(f"{base}{path}", body, {"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=3600).read()


def complete(words: int) -> tuple[int, float]:
    rng = random.Random(words)
    prompt = " ".join(rng.choice(WORDS) for _ in range(words))
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    t0 = time.time()
    out = json.loads(post("/v1/completions", body))
    return out["usage"]["prompt_tokens"], time.time() - t0


# Calibrate words per token on a short prompt (outside the profile), then warm the shape.
tok, _ = complete(2000)
words = int(2000 * n_tokens / tok)
complete(words)
post("/start_profile")
tok, dt = complete(words + 1)  # a different prompt
post("/stop_profile")
print(f"profiled prefill: {tok} prompt tokens, {dt * 1e3:.0f} ms to the first token "
      f"({tok / dt:.0f} tokens/s)")
