"""Prefix caching on V4.1 (PREFIX_CACHING=1): hit against cold outputs, then agentic turns.

1. Hit vs cold. Each prompt is a long few-shot document (GSM8K test questions with their
   worked answers, ~CTX tokens) and one more question. Each is sent three times, greedy,
   with logprobs: with a fresh cache_salt (hashes match nothing, so a cold prefill),
   without salt (cold too, and it fills the cache), and without salt again (the hit).
   Greedy output is not repeatable here (two cold runs can part at the first token, job
   23908822), so the hit is judged against the two cold runs' spread, over all prompts:
   the first token's top-5 logprobs, the tokens shared from the start, and accuracy.
2. Agentic turns. Conversations that resend their growing context each turn (document,
   question, the model's answer, next question, ...): per turn the prompt tokens, cached
   tokens, TTFT and whether the answer is right; the last turn is also sent cold.
3. DSpark acceptance length of cold and hit requests, from /metrics.

    python prefix_cache_check.py http://127.0.0.1:PORT MODEL GSM8K_PARQUET [CTX] [PROMPTS] [TURNS]
"""

import json
import re
import sys
import time
import urllib.request
import uuid

import pandas as pd

base, model, parquet = sys.argv[1:4]
CTX = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
PROMPTS = int(sys.argv[5]) if len(sys.argv) > 5 else 24
TURNS = int(sys.argv[6]) if len(sys.argv) > 6 else 6
CHARS_PER_TOKEN = 3.6  # GSM8K text with this tokenizer, roughly; actual counts are printed

df = pd.read_parquet(parquet)
SHOTS = [f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(df.question[:500], df.answer[:500])]
QUESTIONS = list(zip(df.question[500:], df.answer[500:]))


def document(start: int, tokens: int) -> str:
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


def spec_counters() -> tuple[float, float]:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=60).read().decode()
    acc = drafts = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts += float(line.rsplit(" ", 1)[1])
    return acc, drafts


def complete(prompt: str, max_tokens: int, salt: str | None = None) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": 5, "stop": ["\n\nQuestion:"], "stream": True,
            "stream_options": {"include_usage": True}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(f"{base}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    acc0, drafts0 = spec_counters()
    t0 = time.time()
    ttft, text, tokens, logprobs, top, usage = None, [], [], [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
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
                lp = choice.get("logprobs") or {}
                tokens += lp.get("tokens") or []
                logprobs += lp.get("token_logprobs") or []
                top += lp.get("top_logprobs") or []
    acc1, drafts1 = spec_counters()
    details = usage.get("prompt_tokens_details") or {}
    return {"text": "".join(text), "tokens": tokens, "logprobs": logprobs, "ttft": ttft,
            "first": top[0] if top else {},
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached": details.get("cached_tokens"),
            "accepted": acc1 - acc0, "drafts": drafts1 - drafts0}


def agree(a: dict, b: dict) -> tuple[int, int, float]:
    """Tokens the two outputs share from the start, the shorter length, and the largest
    logprob difference over the shared tokens."""
    n = 0
    for x, y in zip(a["tokens"], b["tokens"]):
        if x != y:
            break
        n += 1
    dlp = max((abs(x - y) for x, y in zip(a["logprobs"][:n], b["logprobs"][:n])
               if x is not None and y is not None), default=0.0)
    return n, min(len(a["tokens"]), len(b["tokens"])), dlp


def first_token_gap(a: dict, b: dict) -> float:
    """Mean |logprob difference| of the first token's top-5 candidates both runs list."""
    shared = [t for t in a["first"] if t in b["first"]]
    return sum(abs(a["first"][t] - b["first"][t]) for t in shared) / max(len(shared), 1)


def mean(xs) -> float:
    return sum(xs) / max(len(xs), 1)


def acceptance(runs: list[dict]) -> str:
    acc = sum(r["accepted"] for r in runs)
    drafts = sum(r["drafts"] for r in runs)
    return f"{1 + acc / drafts:.2f}" if drafts else "n/a"


complete("Warm-up.", 8)
print(f"=== 1. hit vs cold: {PROMPTS} prompts of ~{CTX} tokens ===", flush=True)
colds, hits = [], []
stats = {"cc": [], "ch": [], "gap_cc": [], "gap_ch": [], "right_cold": 0, "right_hit": 0}
for k in range(PROMPTS):
    q, a = QUESTIONS[k]
    prompt = document(37 * k, CTX) + f"Question: {q}\nAnswer:"
    cold_a = complete(prompt, 192, salt=uuid.uuid4().hex)
    cold_b = complete(prompt, 192)
    hit = complete(prompt, 192)
    colds += [cold_a, cold_b]
    hits.append(hit)
    n_cc, len_cc, _ = agree(cold_a, cold_b)
    n_ch, len_ch, _ = agree(cold_b, hit)
    g_cc, g_ch = first_token_gap(cold_a, cold_b), first_token_gap(cold_b, hit)
    stats["cc"].append(n_cc)
    stats["ch"].append(n_ch)
    stats["gap_cc"].append(g_cc)
    stats["gap_ch"].append(g_ch)
    stats["right_cold"] += predicted(cold_b["text"]) == gold(a)
    stats["right_hit"] += predicted(hit["text"]) == gold(a)
    print(f"prompt {k}: {cold_b['prompt_tokens']} tokens, cached {hit['cached']}; "
          f"TTFT cold {cold_a['ttft']:.2f}/{cold_b['ttft']:.2f} s, hit {hit['ttft']:.2f} s; "
          f"shared tokens cold-cold {n_cc}/{len_cc}, cold-hit {n_ch}/{len_ch}; "
          f"first-token gap cold-cold {g_cc:.3f}, cold-hit {g_ch:.3f}; "
          f"answers {predicted(cold_b['text'])}/{predicted(hit['text'])} gold {gold(a)}",
          flush=True)
print(f"over {PROMPTS} prompts: shared tokens from the start, mean cold-cold "
      f"{mean(stats['cc']):.1f}, cold-hit {mean(stats['ch']):.1f}; first-token top-5 gap "
      f"cold-cold {mean(stats['gap_cc']):.3f}, cold-hit {mean(stats['gap_ch']):.3f}; "
      f"right cold {stats['right_cold']}/{PROMPTS}, hit {stats['right_hit']}/{PROMPTS}")
# A hit that reads wrong state shifts the first token's distribution and loses answers;
# noise does neither beyond the cold runs' own spread.
fails = []
if mean(stats["gap_ch"]) > 2 * mean(stats["gap_cc"]) + 0.05:
    fails.append("first-token logprobs")
if stats["right_hit"] < stats["right_cold"] - max(2, PROMPTS // 10):
    fails.append("accuracy")
if mean(stats["ch"]) < 0.5 * mean(stats["cc"]):
    fails.append("shared tokens")
print(f"DSpark acceptance length: cold {acceptance(colds)}, hit {acceptance(hits)}")

print(f"=== 2. agentic: 2 conversations x {TURNS} turns from ~{CTX // 2} tokens ===",
      flush=True)
right = total = 0
turn_runs = []
for conv in range(2):
    context = document(1000 + 211 * conv, CTX // 2)
    for t in range(TURNS):
        q, a = QUESTIONS[100 + conv * TURNS + t]
        prompt = context + f"Question: {q}\nAnswer:"
        r = complete(prompt, 384)
        turn_runs.append(r)
        good = predicted(r["text"]) == gold(a)
        right += good
        total += 1
        print(f"conv {conv} turn {t}: {r['prompt_tokens']} tokens, cached {r['cached']}, "
              f"TTFT {r['ttft']:.2f} s, {len(r['tokens'])} out, "
              f"{'right' if good else 'WRONG'} ({predicted(r['text'])} vs {gold(a)})",
              flush=True)
        context = prompt + r["text"] + "\n\n"
    cold = complete(prompt, 384, salt=uuid.uuid4().hex)
    n, length, dlp = agree(cold, r)
    print(f"conv {conv} last turn cold: TTFT {cold['ttft']:.2f} s against {r['ttft']:.2f} s "
          f"with the cache; same tokens {n}/{length} (max dlogprob {dlp:.3f})", flush=True)
print(f"agentic answers right: {right}/{total}; DSpark acceptance length "
      f"{acceptance(turn_runs)}")
print("PREFIX_CHECK", f"FAILED: {', '.join(fails)}" if fails else "OK")
sys.exit(1 if fails else 0)
