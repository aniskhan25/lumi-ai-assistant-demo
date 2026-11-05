"""CPU self-checks of reference_torch/ against brute-force definitions (run in the container)."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "reference_torch"))
import kernel as K  # noqa: E402
from fast_hadamard_transform import hadamard_transform  # noqa: E402

torch.manual_seed(0)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


n = 16
H = torch.tensor([[1.0]])
while H.shape[0] < n:
    H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
x = torch.randn(3, n)
check("hadamard == x @ H * scale", torch.allclose(hadamard_transform(x, 0.25), x @ H * 0.25, atol=1e-5))

v = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -0.75, -5])
want = [0, 0, .5, 1, 1, 1, 1.5, 2, 2, 2, 3, 4, 4, 4, 6, -1, -4]
got = K._from_e2m1_codes(K._to_e2m1_codes(v)).tolist()
check("e2m1 round-to-nearest-even", got == want, str(got))

x = torch.randn(4, 64) * 3
q, s = K.fp4_act_quant(x.clone())
deq = K._unpack_fp4(q) * s.float().repeat_interleave(32, -1)
y = x.clone()
K.fp4_act_quant(y, inplace=True)
check("fp4 packed path == inplace path", torch.equal(deq, y), f"rel err {((deq - x).norm() / x.norm()).item():.3f}")

a = torch.randn(5, 256)
w = torch.randn(384, 256)
aq, a_s = K.act_quant(a, 128, "ue8m0", torch.float32)
ws = w.abs().reshape(3, 128, 2, 128).amax((1, 3)) / 448
wq = (w / ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
torch.set_default_dtype(torch.float32)
c = K.fp8_gemm(aq, a_s, wq, ws)
err = ((c - a @ w.t()).norm() / (a @ w.t()).norm()).item()
check("fp8 act_quant + fp8_gemm ~ fp32 matmul", err < 0.06, f"rel err {err:.4f}")

q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
kv = torch.randn(1, 5, 8, dtype=torch.bfloat16)
sink = torch.randn(3)
idx = torch.arange(5, dtype=torch.int32).expand(1, 2, 5).clone()
o = K.sparse_attn(q, kv, sink, idx, 0.3).float()
sc = torch.einsum("bshd,bnd->bshn", q.float(), kv.float()) * 0.3
p = torch.exp(sc) / (torch.exp(sc).sum(-1, keepdim=True) + torch.exp(sink)[None, None, :, None])
check("sparse_attn == dense softmax with sink", torch.allclose(o, torch.einsum("bshn,bnd->bshd", p, kv.float()), atol=3e-2))
idx2 = idx.clone()
idx2[..., 3:] = -1
o2 = K.sparse_attn(q, kv, sink, idx2, 0.3).float()
p2 = torch.exp(sc[..., :3]) / (torch.exp(sc[..., :3]).sum(-1, keepdim=True) + torch.exp(sink)[None, None, :, None])
check("sparse_attn ignores -1 rows", torch.allclose(o2, torch.einsum("bshn,bnd->bshd", p2, kv[:, :3].float()), atol=3e-2))

pre, post, comb = K.hc_split_sinkhorn(torch.randn(1, 2, 24), torch.ones(3), torch.zeros(24))
check("sinkhorn ~doubly stochastic", abs(comb.sum(-1).mean().item() - 1) < 1e-3 and abs(comb.sum(-2).mean().item() - 1) < 1e-3)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
