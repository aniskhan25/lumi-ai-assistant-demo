"""Split the GPU time of a vLLM torch-profiler trace (one rank) by model component.

Each GPU kernel is linked to the host call that launched it (`correlation`), and that call
to the Python frames open on its thread at the time (traces taken with_stack=True). The
innermost frame matching a rule below names the component. Run on an eager trace: inside a
CUDA graph, kernels cannot be traced back to Python.

    python analyze_trace.py TRACE.json.gz [COMPONENT ...]

Each COMPONENT named (a prefix, e.g. "MoE") also gets every kernel listed with its calls per
step and time per call.
"""

import collections
import gzip
import json
import sys

# (component, substring of a Python frame name), innermost frame wins; first rule wins.
RULES = [
    ("hyper-connections (mHC)", "kernels/mhc/"),
    ("hyper-connections (mHC)", "layers/mhc.py"),
    ("hyper-connections (mHC)", "mhc_fix.py"),
    ("sparse indexer", "sparse_attn_indexer"),
    ("sparse indexer", "paged_logits.py"),
    ("compressor", "deepseek_v4/compressor.py"),
    ("compressor", "fused_compress_quant_cache"),
    ("sparse attention", "rocm_sparse_attn"),
    ("sparse attention", "_rocm_sparse_attn"),
    ("MoE router guard", "router_guard.py"),
    ("MoE experts", "fused_moe/"),
    ("all-reduce", "communicator"),
    ("all-reduce", "parallel_state.py"),
    # With DSV4_BF16_LINEAR, fp8_utils.py is only the FP8 activation rounding (W8A8 if off)
    ("dense linear: FP8 act rounding / W8A8", "fp8_utils.py"),
    ("dense linear: FP8 act rounding / W8A8", "w8a8"),
    ("dense linear: FP8 act rounding / W8A8", "fp8_round"),
    ("dense linear: bf16 GEMM", "bf16_linear.py"),
    ("linear (other)", "layers/linear.py"),
    ("norms / rope / quant", "layernorm.py"),
    ("norms / rope / quant", "rotary_embedding"),
    ("norms / rope / quant", "input_quant_fp8"),
    ("attention (other)", "deepseek_v4/attention.py"),
    ("embedding / head / sampler", "sampler"),
    ("embedding / head / sampler", "logits_processor"),
    ("embedding / head / sampler", "vocab_parallel_embedding"),
    ("decoder layer (other)", "deepseek_v4/"),
]

events = json.load(gzip.open(sys.argv[1]))["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
launches = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
frames = collections.defaultdict(list)  # tid -> [(start, end, name)]
for e in events:
    if e.get("cat") == "python_function" and e.get("ph") == "X":
        frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))


def classify(stack) -> str:
    for frame in reversed(stack):  # innermost first
        for comp, pattern in RULES:
            if pattern in frame:
                return comp
    return "other"


# One sweep per thread: walk launches in time order, keeping the stack of open frames.
launch_comp = {}
by_tid = collections.defaultdict(list)
for corr, e in launches.items():
    by_tid[e["tid"]].append((e["ts"], corr))
for tid, ls in by_tid.items():
    fl = sorted(frames.get(tid, []))
    i, stack = 0, []
    for ts, corr in sorted(ls):
        while i < len(fl) and fl[i][0] <= ts:
            while stack and stack[-1][1] < fl[i][0]:
                stack.pop()
            stack.append(fl[i])
            i += 1
        while stack and stack[-1][1] < ts:
            stack.pop()
        launch_comp[corr] = classify([f[2] for f in stack])


def component(kernel) -> str:
    name = kernel["name"]
    if "nccl" in name.lower() or "rccl" in name.lower():
        return "all-reduce"
    return launch_comp.get(kernel.get("args", {}).get("correlation"), "unattributed")


per = collections.Counter()
names = collections.defaultdict(collections.Counter)
calls = collections.defaultdict(collections.Counter)
for k in kernels:
    c = component(k)
    per[c] += k["dur"]
    names[c][k["name"][:70]] += k["dur"]
    calls[c][k["name"][:70]] += 1

steps = sum(1 for e in events if e.get("cat") == "python_function"
            and e["name"].endswith(": execute_model") and "gpu_model_runner" in e["name"])
busy = sum(per.values())
span = (max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)) if kernels else 0
print(f"{len(kernels)} kernels, {steps} model steps, GPU busy {busy / 1e3:.1f} ms "
      f"of {span / 1e3:.1f} ms span ({100 * busy / max(span, 1):.0f}%)")
print(f"{'component':<28} {'ms/step':>8} {'share':>6}   top kernels")
for comp, us in per.most_common():
    top = ", ".join(f"{n} {d / max(us, 1):.0%}" for n, d in names[comp].most_common(2))
    print(f"{comp:<28} {us / 1e3 / max(steps, 1):8.2f} {us / busy:6.1%}   {top}")

for want in sys.argv[2:]:
    for comp in [c for c in per if c.startswith(want)]:
        print(f"\n{comp}: kernel, ms/step, calls/step, us/call")
        for n, d in names[comp].most_common():
            c = calls[comp][n]
            print(f"  {n:<70} {d / 1e3 / max(steps, 1):7.2f} {c / max(steps, 1):7.1f} {d / c:8.1f}")
