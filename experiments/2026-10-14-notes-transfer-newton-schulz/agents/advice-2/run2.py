"""Batched Newton-Schulz orthogonalization on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device const float* norm,
                      device half* x, constant uint& size,
                      uint i [[thread_position_in_grid]]) {
    x[i] = half(g[i] / (norm[i / size] + 1e-7f));
}
kernel void polynomial(device const half* a, device const half* aa,
                       device half* b, constant uint& n,
                       uint i [[thread_position_in_grid]]) {
    uint p = i % (n*n);
    b[i] = half(-4.775f * float(a[i]) + 2.0315f * float(aa[i])
                + (p/n == p%n ? 3.4445f : 0.0f));
}
""")


def kernel(g):
    norm = g.norm(dim=(-2, -1), keepdim=True)
    X = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    _lib.normalize(g, norm, X, g.size(-2)*g.size(-1), threads=g.numel())
    tall = g.size(-2) > g.size(-1)
    n = min(g.size(-2), g.size(-1))
    B = torch.empty((g.size(0), n, n), dtype=torch.float16, device=g.device)
    for _ in range(5):
        A = X.mT @ X if tall else X @ X.mT
        AA = A @ A
        _lib.polynomial(A, AA, B, n, threads=B.numel())
        X = X @ B if tall else B @ X
    return X
