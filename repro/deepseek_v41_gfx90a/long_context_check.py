"""Long-context check: needle retrieval and cold time to first token at long prompts.

For each target length, a unique haystack of GSM8K test questions with worked answers,
with one sentence "The secret passphrase for project <NAME> is <7 digits>." inserted at a
depth (fraction of the haystack), then the question. Each request gets a fresh cache_salt,
so every prefill is cold; streamed, so the time to the first token is measured.

    python long_context_check.py URL MODEL GSM8K_PARQUET [LENGTHS...]   (tokens, default 32k 64k 118k)
"""

import json
import random
import sys
import time
import urllib.request
import uuid

import pandas as pd

base, model, parquet = sys.argv[1:4]
LENGTHS = [int(x) for x in sys.argv[4:]] or [32000, 64000, 118000]
DEPTHS = (0.1, 0.5, 0.9)
CHARS_PER_TOKEN = 3.2  # GSM8K text in the chat template, measured; the real count is printed

df = pd.read_parquet(parquet)
SHOTS = [f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(df.question, df.answer)]
NAMES = ["Aurora", "Basalt", "Cinder", "Dune", "Ember", "Fjord", "Glacier", "Harbor", "Isthmus"]


def haystack(tokens: int, rng: random.Random) -> str:
    out, n = [], 0
    while n < tokens * CHARS_PER_TOKEN:
        out.append(rng.choice(SHOTS))
        n += len(out[-1])
    return "".join(out)


def ask(prompt: str) -> tuple[str, float, int]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "cache_salt": uuid.uuid4().hex}
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(delta)
    return "".join(text), ttft or float("nan"), usage.get("prompt_tokens", 0)


ok = total = 0
for length in LENGTHS:
    for i, depth in enumerate(DEPTHS):
        rng = random.Random(length * 10 + i)
        name, code = NAMES[(length // 1000 + i) % len(NAMES)], str(rng.randint(1_000_000, 9_999_999))
        hay = haystack(length, rng)
        cut = hay.rfind("\n\n", 0, int(len(hay) * depth)) + 2
        prompt = (hay[:cut] + f"The secret passphrase for project {name} is {code}.\n\n" + hay[cut:]
                  + f"\nWhat is the secret passphrase for project {name}? Answer with the number only.")
        try:
            answer, ttft, n = ask(prompt)
        except Exception as e:  # e.g. over --max-model-len
            answer, ttft, n = f"ERROR {e}"[:80], float("nan"), 0
        hit = code in answer
        ok += hit
        total += 1
        print(f"~{length // 1000}k (prompt {n} tokens), needle at {int(depth * 100)} %: "
              f"{'found' if hit else 'MISSED'} ({answer.strip()[:40]!r}), TTFT {ttft:.1f} s", flush=True)
print(f"LONG_CONTEXT needles {ok}/{total}")
sys.exit(0 if ok == total else 1)
