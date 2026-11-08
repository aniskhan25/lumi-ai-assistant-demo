"""Does vLLM's fused mHC path (with the Triton big-fuse fix) stay correct under CUDA graphs?

Serving V4-Flash with MHC_FIX=triton gives a NaN logprob on one probe prompt with CUDA
graphs (jobs 23540919, 23541194) but not eager (23542348 with per-call checks, 23542631
without). On one GCD, with every layer's real hc parameters:

  ops    each fused-path op (first-layer pre, fused post+pre, final post, hc_head) captured
         alone per batch size, replayed with new inputs, vs the same op eager on those inputs
  chain  the whole fused mHC chain of a decode step (pre, then fused post+pre twice per
         layer, final post, hc_head; attention and MoE replaced by fixed elementwise
         stand-ins), one graph per batch size in a shared pool captured largest first as
         vLLM does, replayed in random order with new inputs, vs eager and vs the torch
         reference chain (torch pre/post, Triton hc_head: the path that serves correctly)

    python test_mhc_graph.py <DeepSeek-V4-Flash snapshot dir>
"""

import json
import pathlib
import random
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401,E402  registers *_tilelang ops
import vllm.model_executor.kernels.mhc.triton  # noqa: F401,E402  registers hc_head_triton
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch  # noqa: E402

from lumi_dsv4_debug import mhc_fix  # noqa: E402

mhc_fix.install("triton")
snapshot = pathlib.Path(sys.argv[1])
config = json.loads((snapshot / "config.json").read_text())
HC, H, L = config["hc_mult"], config["hidden_size"], config["num_hidden_layers"]
RMS_EPS, HC_EPS = config["rms_norm_eps"], config["hc_eps"]
ARGS = (RMS_EPS, HC_EPS, HC_EPS, 2.0, config["hc_sinkhorn_iters"])
SIZES = [512, 256, 128, 64, 48, 32, 24, 16, 8, 4, 2, 1]  # vLLM captures largest first
dev = torch.device("cuda")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]


def param(name: str) -> torch.Tensor:
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().to(dev)


layers = [{k: [param(f"layers.{i}.hc_{k}_{p}") for p in ("fn", "scale", "base")]
           for k in ("attn", "ffn")} for i in range(L)]
head = [param(f"hc_head_{p}") for p in ("fn", "scale", "base")]
gen = torch.Generator(device=dev).manual_seed(1)
stand_in = [[torch.randn(H, device=dev, generator=gen) * 0.5 for _ in range(2)] for _ in range(L)]
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  hc={HC} hidden={H} layers={L}")


def layer_stand_in(li, w):
    """Attention / MoE stand-in: RMS-normalised layer input times a fixed per-channel vector,
    so the layer output stays bounded like a real one whatever the residual grows to."""
    li = li.float()
    return (li * torch.rsqrt(li.square().mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)


# --- the two chains -------------------------------------------------------------------
def fused_chain(emb):
    """Fused TileLang path as DeepseekV4Model.forward + _forward_fused_post_pre run it."""
    residual = emb.unsqueeze(-2).repeat(1, HC, 1)
    post, res, x = None, None, None
    for i, layer in enumerate(layers):
        for j, kind in enumerate(("attn", "ffn")):
            if x is None:
                post, res, li = torch.ops.vllm.mhc_pre_tilelang(
                    residual, *layer[kind], *ARGS, 1, None, 1e-6)
            else:
                residual, post, res, li = torch.ops.vllm.mhc_fused_post_pre_tilelang(
                    x, residual, post, res, *layer[kind], *ARGS, 1, 1, None, 1e-6)
            x = layer_stand_in(li, stand_in[i][j])
    hidden = torch.ops.vllm.mhc_post_tilelang(x, residual, post, res)
    return torch.ops.vllm.hc_head_fused_kernel_tilelang(hidden, *head, RMS_EPS, HC_EPS)


def reference_chain(emb):
    """The path that serves V4 correctly (DSV4_TORCH_MHC=1): torch pre/post, Triton head."""
    hidden = emb.unsqueeze(-2).repeat(1, HC, 1)
    for i, layer in enumerate(layers):
        for j, kind in enumerate(("attn", "ffn")):
            post, res, li = mhc_pre_torch(hidden, *layer[kind], *ARGS)
            x = layer_stand_in(li, stand_in[i][j])
            hidden = mhc_post_torch(x, hidden, post, res)
    out = torch.empty(hidden.shape[0], H, dtype=torch.bfloat16, device=dev)
    torch.ops.vllm.hc_head_triton(hidden, *head, out, H, RMS_EPS, HC_EPS, HC)
    return out


def rel(got, want) -> float:
    got, want = got.float(), want.float()
    if not torch.isfinite(got).all():
        return float("inf")
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def new_inputs(t: int, scale: float, outliers: bool) -> torch.Tensor:
    e = torch.randn(t, H, device=dev) * scale
    if outliers:
        e[:, torch.randint(0, H, (8,), device=dev)] *= 300
    return e.to(torch.bfloat16)


failures = []

# --- ops: each op captured alone -------------------------------------------------------
print("\nops: captured alone, replayed with new inputs (rel err vs same op eager; FAIL > 1e-3)")
p_attn, p_ffn = layers[5]["attn"], layers[5]["ffn"]
for t in (1, 4, 16, 17, 32, 64):
    static = {"residual": torch.zeros(t, HC, H, dtype=torch.bfloat16, device=dev),
              "x": torch.zeros(t, H, dtype=torch.bfloat16, device=dev),
              "post": torch.zeros(t, HC, 1, device=dev),
              "res": torch.zeros(t, HC, HC, device=dev)}

    def fill():
        static["residual"].copy_(torch.randn(t, HC, H, device=dev).to(torch.bfloat16))
        static["x"].copy_(torch.randn(t, H, device=dev).to(torch.bfloat16))
        post, res, _ = mhc_pre_torch(static["residual"], *p_attn, *ARGS)
        static["post"].copy_(post)
        static["res"].copy_(res)

    ops = {
        "pre": lambda: torch.ops.vllm.mhc_pre_tilelang(static["residual"], *p_attn, *ARGS, 1, None, 1e-6),
        "fused": lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(
            static["x"], static["residual"], static["post"], static["res"], *p_ffn, *ARGS, 1, 1, None, 1e-6),
        "post": lambda: torch.ops.vllm.mhc_post_tilelang(
            static["x"], static["residual"], static["post"], static["res"]),
        "head": lambda: torch.ops.vllm.hc_head_fused_kernel_tilelang(
            static["residual"], *head, RMS_EPS, HC_EPS),
    }
    for name, op in ops.items():
        fill()
        for _ in range(2):
            op()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_out = op()
        graph_out = graph_out if isinstance(graph_out, tuple) else (graph_out,)
        worst = 0.0
        for _ in range(3):
            fill()
            g.replay()
            eager = op()
            eager = eager if isinstance(eager, tuple) else (eager,)
            worst = max([worst] + [rel(a, b) for a, b in zip(graph_out, eager)])
        bad = worst > 1e-3
        if bad:
            failures.append(f"op {name} T={t}")
        print(f"  {'FAIL' if bad else 'ok  '} {name:<6} T={t:<3} worst rel err {worst:.1e}")

# --- chain: whole decode-step mHC chain, vLLM-style capture -------------------------------
print("\nchain: one graph per size, shared pool, largest first")
static_emb = torch.zeros(max(SIZES), H, dtype=torch.bfloat16, device=dev)
for t in SIZES:  # warm up every shape (JIT) before any capture, as vLLM's profile run does
    fused_chain(static_emb[:t])
torch.cuda.synchronize()
pool = torch.cuda.graph_pool_handle()
graphs, outputs = {}, {}
for t in SIZES:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        outputs[t] = fused_chain(static_emb[:t])
    graphs[t] = g
print(f"  captured {len(graphs)} graphs")

rng = random.Random(0)
print("  rel err of graph replay vs eager fused / vs torch reference (FAIL: non-finite or > 2e-2)")
for trial in range(24):
    t = rng.choice(SIZES)
    n = rng.randint(max(1, t // 2), t)  # real tokens; rows n..t keep stale data as padding
    scale, outliers = rng.choice([(0.02, False), (1.0, False), (0.02, True), (5.0, True)])
    static_emb[:n].copy_(new_inputs(n, scale, outliers))
    graphs[t].replay()
    torch.cuda.synchronize()
    got = outputs[t][:n].clone()
    eager = fused_chain(static_emb[:n])
    ref = reference_chain(static_emb[:n])
    e_eager, e_ref, e_eref = rel(got, eager), rel(got, ref), rel(eager, ref)
    bad = not torch.isfinite(got).all() or e_eager > 2e-2 or e_ref > 2e-2
    if bad:
        failures.append(f"chain T={t} n={n} scale={scale} outliers={outliers}")
    print(f"  {'FAIL' if bad else 'ok  '} graph T={t:<3} real {n:<3} scale {scale:<4} outliers "
          f"{outliers!s:<5}  graph/eager {e_eager:.1e}  graph/ref {e_ref:.1e}  (eager/ref {e_eref:.1e})"
          f"  nonfinite rows {int((~torch.isfinite(got.float())).any(-1).sum())}")

# --- poison: are the kernels row-local? --------------------------------------------------
# A padded graph batch runs T rows of which only n are real; the padding rows hold whatever
# the buffers held before, which in serving can be NaN/inf. A row-local kernel keeps that
# out of the real rows.
print("\npoison: rows n..T of every input NaN/inf; real rows vs the op on n rows only "
      "(FAIL: non-finite real rows or rel err > 1e-2)")
for t, n in ((2, 1), (4, 3), (8, 5), (16, 9), (17, 16), (24, 17), (32, 25), (64, 33)):
    residual = torch.randn(t, HC, H, device=dev).to(torch.bfloat16)
    x = torch.randn(t, H, device=dev).to(torch.bfloat16)
    post, res, _ = mhc_pre_torch(residual, *p_attn, *ARGS)
    clean = {
        "pre": lambda r, x, p, c: torch.ops.vllm.mhc_pre_tilelang(r, *p_attn, *ARGS, 1, None, 1e-6),
        "fused": lambda r, x, p, c: torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, r, p, c, *p_ffn, *ARGS, 1, 1, None, 1e-6),
        "post": lambda r, x, p, c: torch.ops.vllm.mhc_post_tilelang(x, r, p, c),
        "head": lambda r, x, p, c: torch.ops.vllm.hc_head_fused_kernel_tilelang(r, *head, RMS_EPS, HC_EPS),
    }
    for poison in (float("nan"), float("inf")):
        args = [a.clone() for a in (residual, x, post, res)]
        for a in args:
            a[n:] = poison
        for name, op in clean.items():
            want = op(*(a[:n].contiguous() for a in (residual, x, post, res)))
            got = op(*args)
            want = want if isinstance(want, tuple) else (want,)
            got = got if isinstance(got, tuple) else (got,)
            nonfinite = sum(int((~torch.isfinite(g[:n].float().reshape(n, -1))).any(-1).sum())
                            for g in got)
            err = max(rel(g[:n], w) for g, w in zip(got, want))  # inf if non-finite
            bad = nonfinite > 0 or err > 1e-2
            if bad:
                failures.append(f"poison {name} T={t} n={n} {poison}")
            print(f"  {'FAIL' if bad else 'ok  '} {name:<6} T={t:<3} real {n:<3} padding {poison!s:<4} "
                  f"non-finite real rows {nonfinite}  worst rel err {err:.1e}")

print("\nFAILED:", failures or "none")
