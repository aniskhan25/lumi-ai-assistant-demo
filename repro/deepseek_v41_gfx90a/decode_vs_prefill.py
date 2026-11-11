"""Localise a decode-time error without a reference implementation.

1. Greedy generation records each decoded token's logprob (decode path).
2. The same prompt + generated tokens are sent back as one prompt with prompt_logprobs:
   the prefill path scores the identical tokens. A correct engine agrees to ~1e-2 nats;
   divergence that grows with position means the decode path (or state carried across
   steps) is wrong.
3. Prefill NLL of a human-written GSM8K solution: a strong model scores well under
   1 nat/token there. High NLL with (2) agreeing means the shared computation is wrong.

python decode_vs_prefill.py --base-url URL --model NAME --gsm8k-parquet P --out diag.json
"""

import argparse
import json
import math

import requests

PROMPTS = [
    "Explain in about 150 words why the sky is blue.",
    "Janet has 3 apples and buys 5 bags with 4 apples each. She eats 2. How many apples "
    "does she have? Solve step by step.",
    "Kirjoita lyhyt kappale Suomen talvesta.",
]


def post(base, path, body):
    r = requests.post(f"{base}{path}", json=body, timeout=1200)
    r.raise_for_status()
    return r.json()


def chat_ids(base, model, messages, add_generation_prompt=True):
    body = {"model": model, "messages": messages,
            "add_generation_prompt": add_generation_prompt}
    return post(base, "/tokenize", body)["tokens"]


def token_id(tok: str) -> int:
    return int(tok.split(":", 1)[1])  # return_tokens_as_token_ids: "token_id:123"


def prefill_logprobs(base, model, ids):
    """logprob of ids[i] given ids[:i], from the prefill path (index 0 is None)."""
    out = post(base, "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
        "prompt_logprobs": 0, "return_tokens_as_token_ids": True})
    plp = out["choices"][0]["prompt_logprobs"]
    res = []
    for i, entry in enumerate(plp):
        if entry is None:
            res.append(None)
            continue
        d = entry.get(str(ids[i])) or entry.get(ids[i])
        res.append(d["logprob"] if d else None)
    return res


def compare(base, model, prompt, max_tokens):
    ids = chat_ids(base, model, [{"role": "user", "content": prompt}])
    gen = post(base, "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
        "logprobs": 0, "return_tokens_as_token_ids": True})["choices"][0]
    lp = gen["logprobs"]
    gen_ids = [token_id(t) for t in lp["tokens"]]
    dec = lp["token_logprobs"]
    pre = prefill_logprobs(base, model, ids + gen_ids)[len(ids):]
    diffs = [abs(a - b) for a, b in zip(dec, pre) if a is not None and b is not None]
    windows = {}
    for start in range(0, len(diffs), 32):
        w = diffs[start:start + 32]
        windows[start] = round(sum(w) / len(w), 4)
    first_big = next((i for i, d in enumerate(diffs) if d > 0.5), None)
    return {
        "prompt": prompt, "prompt_tokens": len(ids), "generated": len(gen_ids),
        "text": gen["text"][:600],
        "decode_nll": round(-sum(dec) / len(dec), 4) if dec else None,
        "prefill_nll_of_same_tokens": round(-sum(p for p in pre if p is not None)
                                            / max(1, len(pre)), 4),
        "mean_abs_diff": round(sum(diffs) / max(1, len(diffs)), 4),
        "max_abs_diff": round(max(diffs, default=0.0), 4),
        "first_diff_gt_0.5": first_big,
        "mean_abs_diff_per_32": windows,
    }


def gold_nll(base, model, question, answer):
    q_ids = chat_ids(base, model, [{"role": "user", "content": question}])
    full = chat_ids(base, model, [{"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}],
                    add_generation_prompt=False)
    lps = prefill_logprobs(base, model, full)[len(q_ids):]
    lps = [x for x in lps if x is not None]
    return {"tokens": len(lps), "nll": round(-sum(lps) / max(1, len(lps)), 4),
            "ppl": round(math.exp(-sum(lps) / max(1, len(lps))), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm8k-parquet")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"compare": [], "gold": []}
    for p in PROMPTS:
        r = compare(a.base_url, a.model, p, a.max_tokens)
        print(f"[diag] gen={r['generated']} decode_nll={r['decode_nll']} "
              f"prefill_nll={r['prefill_nll_of_same_tokens']} mean|d|={r['mean_abs_diff']} "
              f"max|d|={r['max_abs_diff']} first>0.5@{r['first_diff_gt_0.5']} "
              f"per32={r['mean_abs_diff_per_32']}", flush=True)
        print(f"[diag]   {r['text'][:300]!r}", flush=True)
        res["compare"].append(r)
    if a.gsm8k_parquet:
        import pandas as pd

        df = pd.read_parquet(a.gsm8k_parquet).head(5)
        for q, ans in zip(df["question"], df["answer"]):
            g = gold_nll(a.base_url, a.model, q, ans.replace("####", "The answer is"))
            print(f"[gold] tokens={g['tokens']} nll={g['nll']} ppl={g['ppl']}", flush=True)
            res["gold"].append(g)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
