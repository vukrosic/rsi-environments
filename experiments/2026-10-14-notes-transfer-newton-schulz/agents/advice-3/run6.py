"""Muon Newton-Schulz iteration on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void smallmm(device const half* a [[buffer(0)]],
                    device const half* b [[buffer(1)]],
                    device half* out [[buffer(2)]],
                    device half* aux [[buffer(3)]],
                    device const half* extra [[buffer(4)]],
                    constant uint& mode [[buffer(5)]],
                    uint3 group [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half aa[1024];
    threadgroup half bb[1024];
    threadgroup float cc[1024];
    uint row = group.y * 32, col = group.x * 32;
    uint batch = group.z * 65536;
    uint rr = (sg / 2) * 16, cr = (sg % 2) * 16;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint k = 0; k < 256; k += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            aa[i] = a[batch + (row + i / 32) * 256 + k + i % 32];
            bb[i] = b[batch + (k + i / 32) * 256 + col + i % 32];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, aa + rr * 32 + kk, 32);
            simdgroup_load(a1, aa + (rr + 8) * 32 + kk, 32);
            simdgroup_load(b0, bb + kk * 32 + cr, 32);
            simdgroup_load(b1, bb + kk * 32 + cr + 8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, cc + rr * 32 + cr, 32);
    simdgroup_store(c01, cc + rr * 32 + cr + 8, 32);
    simdgroup_store(c10, cc + (rr + 8) * 32 + cr, 32);
    simdgroup_store(c11, cc + (rr + 8) * 32 + cr + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 128) {
        uint r = row + i / 32, c = col + i % 32;
        uint idx = batch + r * 256 + c;
        float v = cc[i];
        if (mode == 1) aux[idx] = half(0.21080563587150372f * v - 2.574272874098163f * float(a[idx]) + (r == c ? 13.238683440917946f : 0.0f));
        if (mode == 2) v = v - 33.75529697507488f * float(extra[idx]) + (r == c ? 40.85264085957168f : 0.0f);
        out[idx] = half(v);
    }
}
""")


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    A = X @ X.mT
    Z = A * 256
    Z2 = torch.empty_like(Z)
    B = torch.empty_like(Z)
    P = torch.empty_like(Z)
    _lib.smallmm(Z, Z, Z2, B, Z, 1, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.smallmm(B, Z2, P, P, Z, 2, threads=(1024, 8, 10), group_size=(128, 1, 1))
    A = P @ (A @ P)
    for i in range(2):
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        B.diagonal(dim1=-2, dim2=-1).add_(3.4445)
        P = B @ P
        if i < 1:
            A = B @ (A @ B)
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
