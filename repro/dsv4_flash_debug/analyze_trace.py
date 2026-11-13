"""Split the GPU time of a vLLM torch-profiler trace (one rank) by model component.

Each GPU kernel is linked to the host call that launched it (`correlation`), and that call
to the Python frames open on its thread at the time (traces taken with_stack=True). The
innermost frame matching a rule below names the component. Run on an eager trace: inside a
CUDA graph, kernels cannot be traced back to Python. On a graph-mode trace (PIECEWISE), the
eager breaks (attention, indexer, ...) are still attributed by Python frame; kernels replayed
from a graph are classified by kernel name (NAME_RULES) and marked "[graph]".

    python analyze_trace.py TRACE.json.gz
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
    ("compressor", "deepseek_v41/compressor.py"),
    ("Engram", "engram"),
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
    ("attention (other)", "deepseek_v41/attention.py"),
    ("attention (other)", "deepseek_v41/amd/"),
    ("embedding / head / sampler", "sampler"),
    ("embedding / head / sampler", "logits_processor"),
    ("embedding / head / sampler", "vocab_parallel_embedding"),
    ("decoder layer (other)", "deepseek_v4/"),
    ("decoder layer (other)", "deepseek_v41/"),
    ("decoder layer (other)", "lumi_dsv41/"),
]

# Graph-replayed kernels: (component, lower-case substring of the kernel name), first wins.
NAME_RULES = [
    ("all-reduce", "cross_device_reduce"), ("all-reduce", "allreduce"),
    ("all-reduce", "nccl"), ("all-reduce", "rccl"),
    ("hyper-connections (mHC)", "mhc"), ("hyper-connections (mHC)", "prenorm_gemm"),
    ("hyper-connections (mHC)", "pre_delayed"), ("hyper-connections (mHC)", "_post_kernel"),
    ("hyper-connections (mHC)", "sinkhorn"), ("hyper-connections (mHC)", "big_fuse"),
    ("MoE experts", "matmul_ogs"), ("MoE experts", "bitmatrix"), ("MoE experts", "routing"),
    ("MoE experts", "topk"), ("MoE experts", "moe"), ("MoE experts", "swiglu"),
    ("MoE experts", "_finalize"), ("MoE experts", "combine"),
    ("Engram", "engram"), ("Engram", "hash"),
    ("compressor", "compress"),
    ("sparse indexer", "indexer"), ("sparse indexer", "mqa_logits"),
    ("sparse attention", "sparse_attn"),
    ("dense linear: FP8 act rounding / W8A8", "fp8_round"),
    ("dense linear: FP8 act rounding / W8A8", "quant"),
    ("GEMM (hipBLASLt / rocBLAS)", "cijk"), ("GEMM (hipBLASLt / rocBLAS)", "gemm"),
    ("GEMM (hipBLASLt / rocBLAS)", "wvsplitk"), ("GEMM (hipBLASLt / rocBLAS)", "skinny"),
    ("norms / rope / quant", "rms"), ("norms / rope / quant", "norm"),
    ("norms / rope / quant", "rope"), ("norms / rope / quant", "rotary"),
    ("elementwise / copy", "elementwise"), ("elementwise / copy", "copy"),
    ("elementwise / copy", "fill"), ("elementwise / copy", "cat"),
    ("elementwise / copy", "index"), ("elementwise / copy", "reduce"),
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


graph_launches = {corr for corr, e in launches.items() if "graphlaunch" in e["name"].lower()}


def component(kernel) -> str:
    name = kernel["name"].lower()
    corr = kernel.get("args", {}).get("correlation")
    if corr in graph_launches:
        return "[graph] " + next((c for c, pat in NAME_RULES if pat in name), "other")
    if "nccl" in name or "rccl" in name:
        return "all-reduce"
    return launch_comp.get(corr, "unattributed")


per = collections.Counter()
names = collections.defaultdict(collections.Counter)
for k in kernels:
    c = component(k)
    per[c] += k["dur"]
    names[c][k["name"][:70]] += k["dur"]

steps = sum(1 for e in events if e.get("cat") == "python_function"
            and e["name"].endswith(": execute_model")
            and ("gpu_model_runner" in e["name"] or "gpu/model_runner" in e["name"]))
busy = sum(per.values())
span = (max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)) if kernels else 0
print(f"{len(kernels)} kernels, {steps} model steps, GPU busy {busy / 1e3:.1f} ms "
      f"of {span / 1e3:.1f} ms span ({100 * busy / max(span, 1):.0f}%)")
print(f"{'component':<44} {'ms/step':>8} {'share':>6}   top kernels")
for comp, us in per.most_common():
    top = ", ".join(f"{n} {d / max(us, 1):.0%}" for n, d in names[comp].most_common(3))
    print(f"{comp:<44} {us / 1e3 / max(steps, 1):8.2f} {us / busy:6.1%}   {top}")
