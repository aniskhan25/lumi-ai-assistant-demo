"""Where do the small kernels in a decode trace come from?

For each kernel whose name contains one of the given substrings (default: the torch copy,
elementwise, fill, gather and FP8-rounding kernels of the graph profiles), per decode step:
GPU time and launches, split by the host call that issued them (a kernel replayed from a
CUDA graph correlates with hipGraphLaunch and has no Python stack). For directly launched
ones, the most common Python stacks (needs a trace recorded with stacks).

    python small_kernels.py TRACE.json.gz STEPS [substring ...]
"""

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
for v in frames.values():
    v.sort()


def stack(launch):
    fs = [f for f in frames.get(launch["tid"], ()) if f[0] <= launch["ts"] <= f[1]]
    return tuple(f[2] for f in fs[-8:])


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
    if launch and "Graph" not in api:
        stacks[e["name"][:110]][stack(launch)] += 1

print(f"per step over {steps:.0f} steps: ms, launches, kernel, issued by")
for (name, api), (dur, n) in sorted(rows.items(), key=lambda r: -r[1][0])[:40]:
    print(f"  {dur / 1e3 / steps:7.3f} ms {n / steps:7.1f}x  {name}  [{api}]")
print("\ndirectly launched: top Python stacks (innermost last)")
for name, c in sorted(stacks.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
    print(f"--- {name}  ({sum(c.values()) / steps:.1f} per step)")
    for st, n in c.most_common(3):
        print(f"    {n / steps:6.1f}x  " + " > ".join(s[:70] for s in st[-5:]))
