"""Muon Newton-Schulz iteration on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void normparts(device const float4* g [[buffer(0)]],
                      device float* parts [[buffer(1)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    float4 v = g[group.y * 45056 + group.x * 256 + tid];
    float s = simd_sum(dot(v, v));
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) parts[group.y * 176 + group.x] = s;
    }
}
kernel void normalize(device const float4* g [[buffer(0)]],
                      device const float* parts [[buffer(1)]],
                      device half4* out [[buffer(2)]],
                      uint2 pos [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    float s = simd_sum(tid < 176 ? parts[pos.y * 176 + tid] : 0.0f);
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) sums[0] = 1.0f / (sqrt(s) + 1e-7f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint idx = pos.y * 45056 + pos.x;
    out[idx] = half4(g[idx] * sums[0]);
}
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
    parts = torch.empty((10, 176), device=g.device)
    normalized = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    _lib.normparts(g, parts, threads=(45056, 10), group_size=(256, 1))
    _lib.normalize(g, parts, normalized, threads=(45056, 10), group_size=(256, 1))
    X = normalized.mT
    A = X @ X.mT
    P = torch.baddbmm(A, A, A, beta=-210.94982910791634, alpha=1659.7599164352375)
    P.diagonal(dim1=-2, dim2=-1).add_(11.864154265859419)
    A = P @ (A @ P)
    for i in range(3):
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        B.diagonal(dim1=-2, dim2=-1).add_(3.4445)
        P = B @ P
        if i < 2:
            A = B @ (A @ B)
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
