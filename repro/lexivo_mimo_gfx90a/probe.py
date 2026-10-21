"""Output-correctness probe for a running vLLM server.

Throughput numbers say nothing about whether the model is right: a server emitting
",,,," benchmarks fine. This checks the output itself, greedy and thinking off:

- sanity: short factual prompts with known answers (English and Finnish), with logprobs
  so non-finite values are caught;
- needle: one fact placed before ~2k and ~16k tokens of filler, i.e. far outside the
  128-token sliding window, so only the global-attention layers can carry it;
- gsm8k: the first N test questions, scored on the final number.

Stdlib only, plus pandas for the GSM8K parquet. Writes one JSON with every response.
"""

import argparse
import concurrent.futures
import json
import math
import re
import sys
import time
import urllib.request

SANITY = [
    ("What is the capital of Finland? Answer with one word.", "helsinki"),
    ("What is 17 multiplied by 23? Answer with the number only.", "391"),
    ("Mikä on Suomen pääkaupunki? Vastaa yhdellä sanalla.", "helsinki"),
    ("Mikä Suomen laki säätelee työnantajan ja työntekijän välistä työsopimusta? "
     "Vastaa lain nimellä.", "työsopimuslaki"),
    ("Translate to English: 'Sopimus on pätemätön, jos se on tehty pakottamalla.'",
     "contract"),
]

FILLER = [
    "The committee reviewed the annual report and noted several minor corrections.",
    "Rainfall in the northern districts was slightly above the long-term average.",
    "The archive moved to a new building with climate-controlled storage rooms.",
    "Several volunteers helped catalogue the donated books over the summer.",
    "The ferry timetable changes twice a year to match the tourist season.",
]


def chat(base_url, model, prompt, max_tokens, logprobs=False, timeout=900):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if logprobs:
        body.update(logprobs=True, top_logprobs=1)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except Exception as e:  # a broken server must still yield a report
        return {"text": "", "error": repr(e)[:300], "finish_reason": None,
                "completion_tokens": 0, "prompt_tokens": 0,
                "seconds": round(time.time() - t0, 2), "nonfinite_logprobs": 0,
                "n_logprobs": 0}
    choice = out["choices"][0]
    lps = [t["logprob"] for t in ((choice.get("logprobs") or {}).get("content") or [])]
    return {
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": out["usage"]["completion_tokens"],
        "prompt_tokens": out["usage"]["prompt_tokens"],
        "seconds": round(time.time() - t0, 2),
        "nonfinite_logprobs": sum(1 for x in lps if not math.isfinite(x)),
        "n_logprobs": len(lps),
    }


def looks_garbage(text):
    """The failure modes seen so far: empty, ",,,,", "!!!!", one token repeated."""
    s = text.strip()
    if not s:
        return True
    if re.search(r"(.)\1{15,}", s):
        return True
    alnum = sum(c.isalnum() for c in s)
    return alnum / len(s) < 0.4


def needle_prompt(n_tokens):
    # ~12 tokens per filler sentence.
    lines = [FILLER[i % len(FILLER)] for i in range(n_tokens // 12)]
    return ("Remember this: the reference number of the case file is 48213.\n\n"
            + "\n".join(lines)
            + "\n\nWhat is the reference number of the case file? "
              "Answer with the number only.")


def last_number(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--gsm8k-parquet", required=True)
    p.add_argument("--gsm8k-n", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    report = {"model": a.model, "sanity": [], "needle": [], "gsm8k": {}}

    for prompt, expect in SANITY:
        r = chat(a.base_url, a.model, prompt, 128, logprobs=True)
        r.update(prompt=prompt, expect=expect, ok=expect in r["text"].lower(),
                 garbage=looks_garbage(r["text"]))
        report["sanity"].append(r)
        print(f"[sanity] ok={r['ok']} garbage={r['garbage']} "
              f"nonfinite={r['nonfinite_logprobs']}/{r['n_logprobs']} :: {r['text'][:120]!r}",
              flush=True)

    for n in (2000, 16000):
        r = chat(a.base_url, a.model, needle_prompt(n), 32)
        r.update(target_tokens=n, ok="48213" in r["text"], garbage=looks_garbage(r["text"]))
        report["needle"].append(r)
        print(f"[needle] prompt_tokens={r['prompt_tokens']} ok={r['ok']} :: {r['text'][:80]!r}",
              flush=True)

    import pandas as pd
    df = pd.read_parquet(a.gsm8k_parquet).head(a.gsm8k_n)

    def one(row):
        q, ans = row
        gold = ans.split("####")[-1].strip().replace(",", "")
        prompt = (q + "\n\nSolve the problem step by step. "
                  "End your reply with a line of the form 'Answer: <number>'.")
        r = chat(a.base_url, a.model, prompt, 1024)
        pred = last_number(r["text"])
        try:
            ok = pred is not None and float(pred) == float(gold)
        except ValueError:
            ok = False
        return {"gold": gold, "pred": pred, "ok": ok, "garbage": looks_garbage(r["text"]),
                "completion_tokens": r["completion_tokens"], "text": r["text"]}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        items = list(ex.map(one, zip(df["question"], df["answer"])))
    n = len(items)
    report["gsm8k"] = {
        "n": n,
        "accuracy": round(sum(i["ok"] for i in items) / n, 4),
        "garbage": sum(i["garbage"] for i in items),
        "seconds": round(time.time() - t0, 1),
        "items": items,
    }
    print(f"[gsm8k] n={n} accuracy={report['gsm8k']['accuracy']} "
          f"garbage={report['gsm8k']['garbage']} in {report['gsm8k']['seconds']} s", flush=True)

    summary = {
        "sanity_ok": sum(r["ok"] for r in report["sanity"]),
        "sanity_n": len(report["sanity"]),
        "needle_ok": sum(r["ok"] for r in report["needle"]),
        "nonfinite_logprobs": sum(r["nonfinite_logprobs"] for r in report["sanity"]),
        "garbage_responses": (sum(r["garbage"] for r in report["sanity"] + report["needle"])
                              + report["gsm8k"]["garbage"]),
        "gsm8k_accuracy": report["gsm8k"]["accuracy"],
    }
    report["summary"] = summary
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("PROBE_SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
