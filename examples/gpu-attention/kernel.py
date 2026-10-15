import torch
import triton
import triton.language as tl

@triton.jit
def _attention(Q, K, V, O, T: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    di = tl.arange(0, D)
    nj = tl.arange(0, BN)
    base = tl.program_id(1) * T * D
    q = tl.load(Q + base + mi[:, None] * D + di[None, :], mi[:, None] < T, 0)
    q = q * (D ** -0.5)
    maxv = tl.full((BM,), float('-inf'), tl.float32)
    norm = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, D), 0.0, tl.float32)
    for start in range(0, (tl.program_id(0) + 1) * BM, BN):
        ni = start + nj
        k = tl.load(K + base + ni[None, :] * D + di[:, None], ni[None, :] < T, 0)
        scores = tl.dot(q.to(tl.float16), k.to(tl.float16)) * 1.4426950408889634
        scores = tl.where((mi[:, None] >= ni[None, :]) & (ni[None, :] < T), scores, float('-inf'))
        newmax = tl.maximum(maxv, tl.max(scores, 1))
        alpha = tl.exp2(maxv - newmax)
        prob = tl.exp2(scores - newmax[:, None])
        values = tl.load(V + base + ni[:, None] * D + di[None, :], ni[:, None] < T, 0)
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), values.to(tl.float16))
        norm = norm * alpha + tl.sum(prob, 1)
        maxv = newmax
    out = acc / norm[:, None]
    tl.store(O + base + mi[:, None] * D + di[None, :], out, mi[:, None] < T)

def kernel(q, k, v):
    b, h, t, d = q.shape
    out = torch.empty_like(q)
    _attention[(triton.cdiv(t, 128), b * h)](q, k, v, out, t, d, 128, 64, num_warps=8, num_stages=1)
    return out
