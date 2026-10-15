"""Muon orthogonalization using a polynomial on the smaller Gram matrix."""
import torch

_identity = torch.eye(256, dtype=torch.float16, device="mps").expand(10, -1, -1)

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device const float* norms,
                      device half* out, uint i [[thread_position_in_grid]]) {
    out[i] = half(g[i] / (norms[i/180224] + 1e-7f));
}
kernel void affine(device const half* a, device half* out,
                   constant float& scale, constant float& diagonal,
                   uint i [[thread_position_in_grid]]) {
    uint q = i % 65536;
    out[i] = half(scale * float(a[i]) + (q/256 == q%256 ? diagonal : 0.0f));
}
kernel void poly(device const half* a, device const half* aa, device half* out,
                 uint i [[thread_position_in_grid]]) {
    uint q = i % 65536;
    out[i] = half(2.0315f * float(aa[i]) - 4.7750f * float(a[i]) +
                 (q/256 == q%256 ? 3.4445f : 0.0f));
}
""")

def _affine(a, scale, diagonal):
    out = torch.empty_like(a)
    _lib.affine(a, out, scale, diagonal, threads=a.numel(), group_size=256)
    return out

def kernel(g):
    norms = g.norm(dim=(-2, -1), keepdim=True)
    normalized = torch.empty_like(g, dtype=torch.float16)
    _lib.normalize(g, norms, normalized, threads=g.numel(), group_size=256)
    X = normalized.mT
    A = torch.bmm(X, X.mT)
    S = _affine(A, 128.0, 0.0)
    # Approximation to the first three Newton-Schulz steps on [0, 0.012].
    coeff = (40.839021819581134, -8630.287176555892/128,
             865621.7461404352/128**2, -43113954.891333096/128**3,
             907838639.8434815/128**4)
    P = _affine(S, coeff[4], coeff[3])
    for c in coeff[2::-1]:
        P = torch.baddbmm(_identity, P, S, beta=c)
    A = torch.bmm(torch.bmm(P, A), P.mT)
    for i in range(2):
        T = torch.empty_like(A)
        _lib.poly(A, torch.bmm(A, A), T, threads=A.numel(), group_size=256)
        P = torch.bmm(T, P)
        if i == 0:
            A = torch.bmm(torch.bmm(T, A), T.mT)
    return torch.bmm(P, X).mT
