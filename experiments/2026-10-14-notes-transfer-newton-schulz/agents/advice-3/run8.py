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
    threadgroup half aa[2048];
    threadgroup half bb[1024];
    threadgroup float cc[2048];
    uint row = group.y * 64, col = group.x * 32;
    uint batch = group.z * 65536;
    uint rr = (sg / 2) * 32, cr = (sg % 2) * 16;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f), c20(0.0f), c21(0.0f), c30(0.0f), c31(0.0f);
    for (uint k = 0; k < 256; k += 32) {
        for (uint i = tid; i < 2048; i += 128) {
            aa[i] = a[batch + (row + i / 32) * 256 + k + i % 32];
        }
        for (uint i = tid; i < 1024; i += 128) {
            bb[i] = b[batch + (k + i / 32) * 256 + col + i % 32];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, aa + rr * 32 + kk, 32);
            simdgroup_load(a1, aa + (rr + 8) * 32 + kk, 32);
            simdgroup_load(a2, aa + (rr + 16) * 32 + kk, 32);
            simdgroup_load(a3, aa + (rr + 24) * 32 + kk, 32);
            simdgroup_load(b0, bb + kk * 32 + cr, 32);
            simdgroup_load(b1, bb + kk * 32 + cr + 8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
            simdgroup_multiply_accumulate(c20, a2, b0, c20);
            simdgroup_multiply_accumulate(c21, a2, b1, c21);
            simdgroup_multiply_accumulate(c30, a3, b0, c30);
            simdgroup_multiply_accumulate(c31, a3, b1, c31);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, cc + rr * 32 + cr, 32);
    simdgroup_store(c01, cc + rr * 32 + cr + 8, 32);
    simdgroup_store(c10, cc + (rr + 8) * 32 + cr, 32);
    simdgroup_store(c11, cc + (rr + 8) * 32 + cr + 8, 32);
    simdgroup_store(c20, cc + (rr + 16) * 32 + cr, 32);
    simdgroup_store(c21, cc + (rr + 16) * 32 + cr + 8, 32);
    simdgroup_store(c30, cc + (rr + 24) * 32 + cr, 32);
    simdgroup_store(c31, cc + (rr + 24) * 32 + cr + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 2048; i += 128) {
        uint r = row + i / 32, c = col + i % 32;
        uint idx = batch + r * 256 + c;
        float v = cc[i];
        if (mode == 1) aux[idx] = half(0.21080563587150372f * v - 2.574272874098163f * float(a[idx]) + (r == c ? 13.238683440917946f : 0.0f));
        if (mode == 2) v = v - 33.75529697507488f * float(extra[idx]) + (r == c ? 40.85264085957168f : 0.0f);
        if (mode == 3) v = 2.0315f * v - 4.775f * float(a[idx]) + (r == c ? 3.4445f : 0.0f);
        out[idx] = half(v);
    }
}
""")


def _mm(a, b, mode=0):
    out = torch.empty_like(a)
    _lib.smallmm(a, b, out, out, a, mode, threads=(1024, 4, 10), group_size=(128, 1, 1))
    return out


def kernel(g):
    parts = torch.empty((10, 176), device=g.device)
    normalized = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    _lib.normparts(g, parts, threads=(45056, 10), group_size=(256, 1))
    _lib.normalize(g, parts, normalized, threads=(45056, 10), group_size=(256, 1))
    X = normalized.mT
    A = X @ X.mT
    Z = A * 256
    Z2 = torch.empty_like(Z)
    B = torch.empty_like(Z)
    P = torch.empty_like(Z)
    _lib.smallmm(Z, Z, Z2, B, Z, 1, threads=(1024, 4, 10), group_size=(128, 1, 1))
    _lib.smallmm(B, Z2, P, P, Z, 2, threads=(1024, 4, 10), group_size=(128, 1, 1))
    A = _mm(P, _mm(A, P))
    for i in range(2):
        B = _mm(A, A, 3)
        P = _mm(B, P)
        if i < 1:
            A = _mm(B, _mm(A, B))
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
