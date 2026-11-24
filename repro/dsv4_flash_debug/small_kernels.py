"""Where do the small kernels in a decode trace come from?

For each kernel whose name contains one of the given substrings (default: the torch copy,
elementwise, fill, gather and FP8-rounding kernels of the graph profiles), per decode step:
GPU time and launches, split by the host call that issued them (a kernel replayed from a
CUDA graph correlates with hipGraphLaunch and has no Python stack). For directly launched
ones, the most common Python stacks (needs a trace recorded with stacks).

    python small_kernels.py TRACE.json.gz STEPS [substring ...]
"""

import bisect
import collections
import gzip
import json
import sys

DEFAULT = ["direct_copy", "elementwise_kernel", "copyBuffer", "Memcpy", "FillFunctor",
           "gather", "_fp8_round", "_splitk_sum", "sigmoid", "bfloat16_copy", "CUDAFunctor_add",
           "BinaryFunctor", "AUnaryFunctor", "index_elementwise", "reduce_kernel", "scatter"]

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
steps = float(sys.argv[2])
wants = sys.argv[3:] or DEFAULT
runtime = {e["args"]["correlation"]: e for e in ev
           if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
frames = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "python_function" and e.get("ph") == "X":
        frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
starts = {}
for tid, v in frames.items():
    v.sort()
    starts[tid] = [f[0] for f in v]
SAMPLE = 300  # stacks per kernel name: enough to rank the callers


def stack(launch):
    """Enclosing frames, innermost last: binary search, then walk back over earlier frames
    (bounded; enclosing frames start earlier and end after the launch)."""
    fs, st = frames.get(launch["tid"], []), starts.get(launch["tid"], [])
    i = bisect.bisect_right(st, launch["ts"]) - 1
    out = []
    for j in range(i, max(-1, i - 20000), -1):
        if fs[j][1] >= launch["ts"]:
            out.append(fs[j][2])
            if len(out) == 8:
                break
    return tuple(reversed(out))


rows = collections.defaultdict(lambda: [0.0, 0])
stacks = collections.defaultdict(collections.Counter)
for e in ev:
    if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
        continue
    want = next((w for w in wants if w in e["name"]), None)
    if want is None:
        continue
    launch = runtime.get(e.get("args", {}).get("correlation"))
    api = launch["name"] if launch else "?"
    key = (e["name"][:110], api)
    rows[key][0] += e.get("dur", 0)
    rows[key][1] += 1
    if launch and "Graph" not in api and sum(stacks[e["name"][:110]].values()) < SAMPLE:
        stacks[e["name"][:110]][stack(launch)] += 1

rows_by_name = collections.Counter()
for (name, api), (dur, n) in rows.items():
    rows_by_name[name] += dur
print(f"per step over {steps:.0f} steps: ms, launches, kernel, issued by")
for (name, api), (dur, n) in sorted(rows.items(), key=lambda r: -r[1][0])[:40]:
    print(f"  {dur / 1e3 / steps:7.3f} ms {n / steps:7.1f}x  {name}  [{api}]")
print("\ndirectly launched: top Python stacks (innermost last)")
for name, c in sorted(stacks.items(), key=lambda kv: -rows_by_name.get(kv[0], 0))[:14]:
    tot = sum(c.values())
    print(f"--- {name}  ({tot} launches sampled)")
    for st, n in c.most_common(4):
        print(f"    {100 * n / tot:5.1f} %  " + " > ".join(s[:80] for s in st[-5:]))
