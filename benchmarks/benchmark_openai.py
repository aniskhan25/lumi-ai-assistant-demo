#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def request_json(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_models_endpoint(
    base_url: str,
    startup_wait_s: float,
    poll_interval_s: float,
    request_timeout_s: float,
) -> dict:
    deadline = time.time() + startup_wait_s
    last_error = None
    url = f"{base_url}/models"

    while time.time() < deadline:
        try:
            return request_json("GET", url, None, request_timeout_s)
        except Exception as exc:
            last_error = exc
            remaining = max(0.0, deadline - time.time())
            print(
                f"Waiting for vLLM endpoint {url} "
                f"(remaining {remaining:.0f}s): {exc}"
            )
            time.sleep(poll_interval_s)

    raise RuntimeError(
        f"vLLM endpoint {url} did not become ready within {startup_wait_s:.0f}s. "
        f"Last error: {last_error}"
    )


def model_id_from_models_response(models_body: dict, base_url: str) -> str:
    models = models_body.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/models")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError("Model id missing from /models response")
    return model_id


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
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
        usage = body.get("usage", {})
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
    parser.add_argument("--prompts-file", default="benchmarks/prompts.txt")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--startup-wait-s",
        type=float,
        default=180.0,
        help="Max time to wait for /models before starting benchmark",
    )
    parser.add_argument(
        "--startup-poll-s",
        type=float,
        default=2.0,
        help="Polling interval while waiting for /models",
    )
    parser.add_argument(
        "--output-json",
        default="benchmarks/results/latest_summary.json",
        help="Summary output path on local filesystem",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)

    try:
        models_body = wait_for_models_endpoint(
            args.base_url,
            startup_wait_s=args.startup_wait_s,
            poll_interval_s=args.startup_poll_s,
            request_timeout_s=min(args.timeout, 10.0),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    model = args.model or model_id_from_models_response(models_body, args.base_url)
    print(f"Benchmark model: {model}")
    print(f"Base URL: {args.base_url}")
    print(
        f"Requests: {args.requests}, Concurrency: {args.concurrency}, "
        f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}"
    )

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_one,
                i,
                args.base_url,
                model,
                prompts[i % len(prompts)],
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
            for i in range(args.requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - start_all

    summary = summarize(results, elapsed, args.concurrency)
    summary["model"] = model
    summary["base_url"] = args.base_url
    summary["max_tokens"] = args.max_tokens
    summary["temperature"] = args.temperature

    out_json_dir = os.path.dirname(args.output_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {args.output_json}")

    return 0 if summary["requests_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
