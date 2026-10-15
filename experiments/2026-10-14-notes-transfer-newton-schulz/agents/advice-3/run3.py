"""Muon Newton-Schulz iteration on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void smallmm(device const half* a [[buffer(0)]],
                    device const half* b [[buffer(1)]],
                    device half* out [[buffer(2)]],
                    constant uint& mode [[buffer(3)]],
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
        if (mode) v = 2.0315f * v - 4.775f * float(a[idx]) + (r == c ? 3.4445f : 0.0f);
        out[idx] = half(v);
    }
}
""")


def _mm(a, b, mode=0):
    out = torch.empty_like(a)
    _lib.smallmm(a, b, out, mode, threads=(1024, 8, 10), group_size=(128, 1, 1))
    return out


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    A = X @ X.mT
    P = None
    for i in range(5):
        B = _mm(A, A, 1)
        P = B if P is None else _mm(B, P)
        if i < 4:
            A = _mm(B, _mm(A, B))
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
