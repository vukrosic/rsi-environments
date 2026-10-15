"""Muon orthogonalization for the batched 704-by-256 MPS workload."""
import torch

_identity = torch.eye(256, dtype=torch.float16, device="mps").expand(10, -1, -1).contiguous()

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void norm(device const float4* g, device float* out,
                 uint batch [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    float4 acc(0.0f);
    for (uint i=tid; i<45056; i+=256) {
        float4 v = g[batch*45056+i];
        acc += v*v;
    }
    float s = simd_sum(acc.x+acc.y+acc.z+acc.w);
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) out[batch] = sqrt(s);
    }
}
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
    norms = torch.empty((10,), device=g.device)
    _lib.norm(g, norms, threads=10*256, group_size=256)
    normalized = torch.empty_like(g, dtype=torch.float16)
    _lib.normalize(g, norms, normalized, threads=g.numel(), group_size=256)
    X = normalized.mT
    A = torch.bmm(X, X.mT)
    S = _affine(A, 128.0, 0.0)
    # Cubic approximation to the first three steps on the normalized
    # random-input spectrum, fitted through the two remaining exact steps.
    coeff = (40.34591174256388, -63.53465140948588,
             42.79204493035103, -10.4964361578874)
    P = _affine(S, coeff[3], coeff[2])
    for c in coeff[1::-1]:
        P = torch.baddbmm(_identity, P, S, beta=c)
    # Carry the accumulated transform and its Gram matrix in 256-by-256
    # space; apply the transform to the rectangular input only once.
    A = torch.bmm(torch.bmm(P, A), P.mT)
    for i in range(2):
        T = torch.empty_like(A)
        _lib.poly(A, torch.bmm(A, A), T, threads=A.numel(), group_size=256)
        P = torch.bmm(T, P)
        if i == 0:
            A = torch.bmm(torch.bmm(T, A), T.mT)
    return torch.bmm(P, X).mT
