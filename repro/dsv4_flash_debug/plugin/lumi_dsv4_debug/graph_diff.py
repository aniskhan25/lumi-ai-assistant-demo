"""Graph-vs-eager differential check, per layer (DSV4_GRAPH_DIFF=1, debug).

After every graph replay of a batch of at most MAX_ROWS tokens, the same step runs again
eagerly on the same static inputs (same padded batch, same attention metadata; the KV
cache writes repeat with the same values). Every decoder layer copies its attention
output, MoE output and residual stream into static buffers allocated before capture, so
the copies are replayed with the graph and do not change the graph-pool layout. The
two sets are compared on the real rows (slot_mapping >= 0), and the first site that
differs is logged: that is where graph replay first computes something eager does not.

DSV4_GRAPH_DIFF_TOL (default 1e-2): relative threshold, max |graph - eager| over the
real rows divided by max |eager|. Rank 0 logs; all ranks rerun (the step has collectives).
"""

import os
import sys

import torch

MAX_ROWS = int(os.environ.get("DSV4_GRAPH_DIFF_ROWS", "128"))
TOL = float(os.environ.get("DSV4_GRAPH_DIFF_TOL", "1e-2"))

_bufs: dict[str, torch.Tensor] = {}
_order: list[str] = []
_stats = {"steps": 0, "diverged": 0, "skipped": 0}


def _log(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[lumi_dsv4_debug] graph_diff: {msg}", file=sys.stderr, flush=True)


def _record(name: str, t: torch.Tensor | None) -> None:
    if not isinstance(t, torch.Tensor):
        return
    buf = _bufs.get(name)
    if buf is None:
        # first seen in the profile / warm-up runs, before any capture
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"graph_diff: buffer {name} first seen during capture")
        buf = torch.zeros((MAX_ROWS, *t.shape[1:]), dtype=t.dtype, device=t.device)
        _bufs[name] = buf
        _order.append(name)
    if t.shape[0] <= MAX_ROWS:
        buf[: t.shape[0]].copy_(t)


def _wrap(module, name, pick):
    orig = module.forward

    def forward(*a, **k):
        out = orig(*a, **k)
        _record(name, pick(out))
        return out

    module.forward = forward


def _instrument(model) -> None:
    for i, layer in enumerate(model.layers):
        if not hasattr(layer, "attn"):
            continue
        _wrap(layer.attn, f"L{i:02d}.attn", lambda o: o)
        _wrap(layer.ffn, f"L{i:02d}.ffn", lambda o: o)
        # fused path: (x, residual, post, comb); unfused: (stream, None, None, None)
        _wrap(layer, f"L{i:02d}.stream", lambda o: o[1] if o[1] is not None else o[0])


def _real_rows(num_tokens: int) -> torch.Tensor | None:
    from vllm.forward_context import get_forward_context

    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if not isinstance(md, dict):
        return None
    for m in md.values():
        sm = getattr(m, "slot_mapping", None)
        if isinstance(sm, torch.Tensor) and sm.shape[0] >= num_tokens:
            return sm[:num_tokens] >= 0
    return None


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def install() -> None:
    import vllm.models.deepseek_v4.amd.model as m
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    orig_fwd = m.DeepseekV4Model.forward

    def model_forward(self, input_ids, positions, *a, **k):
        if not getattr(self, "_graph_diff_done", False):
            _instrument(self)
            self._graph_diff_done = True
        out = orig_fwd(self, input_ids, positions, *a, **k)
        _record("final", _first(out))
        return out

    m.DeepseekV4Model.forward = model_forward

    orig_replay = BreakableCUDAGraphWrapper._replay

    def _replay(self, entry, args, kwargs):
        out = orig_replay(self, entry, args, kwargs)
        n = _first(out).shape[0]
        real = _real_rows(n)
        if n > MAX_ROWS or real is None or not _bufs:
            _stats["skipped"] += 1
            return out
        graph = {k: _bufs[k][:n].clone() for k in _order}
        self.runnable(*args, **kwargs)  # eager rerun refills the buffers
        idx = real.nonzero().squeeze(1)
        bad = []
        for k in _order:
            g, e = graph[k].index_select(0, idx).float(), _bufs[k][:n].index_select(0, idx).float()
            scale = e.abs().max().clamp_min(1e-6)
            nonfinite = int((~torch.isfinite(g)).any(dim=tuple(range(1, g.ndim))).sum())
            rel = ((g - e).abs().nan_to_num(float("inf")).max() / scale).item()
            if rel > TOL or nonfinite:
                rows = ((g - e).abs().nan_to_num(float("inf"))
                        .amax(dim=tuple(range(1, g.ndim))) / scale > TOL).nonzero().squeeze(1)
                bad.append((k, rel, nonfinite, idx[rows].tolist()[:8]))
        _stats["steps"] += 1
        if bad:
            _stats["diverged"] += 1
            first = bad[0]
            _log(f"step {_stats['steps']} tokens {n} real {len(idx)} DIVERGED at {first[0]} "
                 f"rel {first[1]:.3g} nonfinite rows {first[2]} rows {first[3]}; "
                 f"{len(bad)} sites, next: "
                 + ", ".join(f"{b[0]} {b[1]:.3g}" for b in bad[1:6]))
        if _stats["steps"] % 500 == 0:
            _log(f"{_stats}")
        return out

    BreakableCUDAGraphWrapper._replay = _replay
