#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def request_json(method: str, url: str, payload: Optional[dict], timeout: float) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_model_id(base_url: str, timeout: float) -> str:
    body = request_json("GET", f"{base_url}/models", None, timeout)
    models = body.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/models")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError("Model id missing from /models response")
    return model_id


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = round((p / 100.0) * (len(values) - 1))
    return sorted(values)[idx]


def run_one(
    request_id: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    try:
        body = request_json("POST", url, payload, timeout)
        elapsed = time.perf_counter() - start
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return {
            "request_id": request_id,
            "ok": True,
            "latency_s": elapsed,
            "prompt": prompt,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "ok": False,
            "latency_s": elapsed,
            "prompt": prompt,
            "error": str(exc),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def summarize(results: list[dict], total_elapsed_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = [r["latency_s"] for r in ok]
    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in ok)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_tokens = sum(r.get("total_tokens", 0) for r in ok)
    success_count = len(ok)
    request_count = len(results)

    summary = {
        "requests_total": request_count,
        "requests_ok": success_count,
        "requests_failed": len(failed),
        "concurrency": concurrency,
        "elapsed_s": total_elapsed_s,
        "throughput_req_s": (success_count / total_elapsed_s) if total_elapsed_s > 0 else 0.0,
        "latency_p50_s": percentile(latencies, 50.0),
        "latency_p95_s": percentile(latencies, 95.0),
        "latency_p99_s": percentile(latencies, 99.0),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "tokens_prompt_total": prompt_tokens,
        "tokens_completion_total": completion_tokens,
        "tokens_total": total_tokens,
        "throughput_tokens_s": (total_tokens / total_elapsed_s) if total_elapsed_s > 0 else 0.0,
        "throughput_completion_tokens_s": (
            completion_tokens / total_elapsed_s
            if total_elapsed_s > 0
            else 0.0
        ),
        "sample_errors": [r.get("error", "") for r in failed[:5]],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple OpenAI-compatible benchmark runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None, help="Optional. If unset, use first model from /models")
    parser.add_argument("--prompts-file", default="benchmarks/prompts_puhti.txt")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="benchmarks/results/latest_summary.json",
        help="Summary output path on local filesystem",
    )
    parser.add_argument(
        "--output-raw-json",
        default="benchmarks/results/latest_raw.json",
        help="Per-request output path on local filesystem",
    )
    args = parser.parse_args()

    if args.requests <= 0:
        raise ValueError("--requests must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    random.seed(args.seed)
    prompts = load_prompts(args.prompts_file)

    model = args.model or get_model_id(args.base_url, args.timeout)
    print(f"Benchmark model: {model}")
    print(f"Base URL: {args.base_url}")
    print(
        f"Requests: {args.requests}, Concurrency: {args.concurrency}, "
        f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}"
    )

    start_all = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i in range(args.requests):
            prompt = random.choice(prompts)
            futures.append(
                pool.submit(
                    run_one,
                    i,
                    args.base_url,
                    model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - start_all

    summary = summarize(results, elapsed, args.concurrency)
    summary["model"] = model
    summary["base_url"] = args.base_url
    summary["max_tokens"] = args.max_tokens
    summary["temperature"] = args.temperature

    out_json_dir = os.path.dirname(args.output_json)
    out_raw_dir = os.path.dirname(args.output_raw_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    if out_raw_dir:
        os.makedirs(out_raw_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_raw_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {args.output_json}")
    print(f"Wrote raw results: {args.output_raw_json}")

    return 0 if summary["requests_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
