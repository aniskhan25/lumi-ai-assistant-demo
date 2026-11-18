"""Where a decode step's wall time goes in a graph-mode torch-profiler trace (one rank).

analyze_trace.py splits GPU time by Python component, which needs an eager trace. Under
piecewise CUDA graphs the question is different: how much of the step is the GPU idle, and
is it idle because the host launched the next kernel late? For every gap between kernels
on the busiest GPU stream, the next kernel's launch call (runtime event with the same
correlation id) says which: launched after the GPU went idle = host-bound gap.

    python graph_trace.py TRACE.json.gz [STEP_KERNEL_SUBSTRING [CALLS_PER_STEP]]

STEP_KERNEL_SUBSTRING names a kernel that runs CALLS_PER_STEP (default 1) times per step, to
count steps; without it the script lists candidates (kernels by call count). All-reduce
kernels spin while they wait for the slowest rank, so their time beyond the fastest calls
is reported apart from compute.
"""

import collections
import gzip
import json
import sys

path = sys.argv[1]
step_key = sys.argv[2] if len(sys.argv) > 2 else None
per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
AR = ("cross_device_reduce", "ncclDevKernel")
with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
    ev = json.load(f)["traceEvents"]

gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
runtime = {e["args"]["correlation"]: e for e in ev
           if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
           and "correlation" in e.get("args", {})}
streams = collections.Counter((e["pid"], e["tid"]) for e in gpu)
print("GPU streams (pid, tid): events", streams.most_common(4))
main = streams.most_common(1)[0][0]
k = sorted((e for e in gpu if (e["pid"], e["tid"]) == main), key=lambda e: e["ts"])
# All streams of that device, for the busy union (side streams overlap the main one)
dev = [e for e in gpu if e["pid"] == main[0]]
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in dev)
busy, cur_s, cur_e = 0.0, *iv[0]
for s, e in iv[1:]:
    if s > cur_e:
        busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
t0, t1 = iv[0][0], max(e for _, e in iv)
span = t1 - t0

steps = None
if step_key:
    n = sum(1 for e in k if step_key in e["name"])
    steps = n / per_step or None
    print(f"steps: {n} launches of a kernel matching {step_key!r} / {per_step} = {n / per_step:.1f}")
div = steps or 1
unit = "per step" if steps else "total"
print(f"window {span / 1e3:.1f} ms, GPU busy {busy / 1e3:.1f} ms ({100 * busy / span:.0f} %)"
      + (f"; per step: wall {span / div / 1e3:.2f} ms, busy {busy / div / 1e3:.2f} ms" if steps else ""))
ar = sorted(e["dur"] for e in dev if any(x in e["name"] for x in AR))
if ar:
    floor = ar[len(ar) // 10]  # 10th percentile call: the transfer itself, little waiting
    print(f"all-reduce: {len(ar) / div:.0f} calls {unit}, {sum(ar) / div / 1e3:.2f} ms {unit}, "
          f"p10 {floor:.1f} us, median {ar[len(ar) // 2]:.1f} us -> waiting for other ranks "
          f"~{(sum(ar) - floor * len(ar)) / div / 1e3:.2f} ms {unit}; compute (busy - all-reduce) "
          f"{(busy - sum(ar)) / div / 1e3:.2f} ms {unit}")

# Gaps on the main stream, split by whether the next kernel was launched late
host_gap, dev_gap, n_host = 0.0, 0.0, 0
by_next = collections.defaultdict(lambda: [0.0, 0])
by_api = collections.defaultdict(lambda: [0.0, 0])
hist = collections.Counter()
end = k[0]["ts"] + k[0]["dur"]
for e in k[1:]:
    gap = e["ts"] - end
    if gap > 2:
        launch = runtime.get(e.get("args", {}).get("correlation"))
        late = launch is not None and launch["ts"] + launch["dur"] > end + 1
        if late:
            host_gap += gap
            n_host += 1
            by_next[e["name"][:90]][0] += gap
            by_next[e["name"][:90]][1] += 1
            by_api[launch["name"]][0] += gap
            by_api[launch["name"]][1] += 1
        else:
            dev_gap += gap
        hist[min(int(gap // 50) * 50, 1000)] += gap
    end = max(end, e["ts"] + e["dur"])
print(f"main-stream gaps: host late {host_gap / div / 1e3:.2f} ms {unit} ({n_host / div:.0f} gaps), "
      f"other {dev_gap / div / 1e3:.2f} ms {unit}")
print("gap time by gap size (us bucket: ms " + unit + "):",
      ", ".join(f"{b}+: {v / div / 1e3:.2f}" for b, v in sorted(hist.items())))
print("\nhost-late gaps by launch API:")
for name, (t, n) in sorted(by_api.items(), key=lambda x: -x[1][0])[:8]:
    print(f"  {t / div / 1e3:7.2f} ms  {n / div:6.0f}x  {name}")
print("\nhost-late gaps by the kernel that waited (top 25):")
for name, (t, n) in sorted(by_next.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  {t / div / 1e3:7.2f} ms  {n / div:6.1f}x  {t / n:7.1f} us  {name}")

agg = collections.defaultdict(lambda: [0.0, 0])
for e in dev:
    agg[e["name"][:110]][0] += e["dur"]
    agg[e["name"][:110]][1] += 1
print(f"\nkernels by GPU time ({unit}; top 45):")
for name, (t, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:45]:
    print(f"  {t / div / 1e3:7.3f} ms  {n / div:7.1f}x  {t / n:7.1f} us  {name}")

api = collections.defaultdict(lambda: [0.0, 0])
for e in runtime.values():
    api[e["name"]][0] += e["dur"]
    api[e["name"]][1] += 1
print(f"\nhost runtime calls ({unit}):")
for name, (t, n) in sorted(api.items(), key=lambda x: -x[1][0])[:12]:
    print(f"  {t / div / 1e3:7.2f} ms  {n / div:8.1f}x  {name}")
if not steps:
    print("\nstep marker candidates (kernels by calls):")
    cnt = collections.Counter(e["name"][:90] for e in k)
    small = [(n, name) for name, n in cnt.items() if n >= 5]
    for n, name in sorted(small)[:25]:
        print(f"  {n:6d}  {name}")
