"""Causal attention using tiled Metal matrix operations."""
import torch
import torch.nn.functional as F

_BM = 64
_BN = 64
_SOURCE = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint BM = __BM__;
constant uint BN = __BN__;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* Y [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half storage[BN * 128];
    threadgroup half* qs = storage;
    threadgroup half* ks = storage;
    threadgroup half* vs = storage + BN * 64;
    uint base = group.z * 1024 * 64;
    uint qstart = group.y * BM;
    for (uint i = tid; i < BM * 64; i += BM * 4)
        qs[i] = half(Q[base + qstart * 64 + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qa[8];
    simdgroup_float8x8 out[8];
    #pragma unroll
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(qa[d], qs + sg * 8 * 64 + d * 8, 64);
        out[d] = simdgroup_float8x8(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = ((lane >> 1) & 3) + (lane >> 4) * 4;
    uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
    uint qrow = qstart + sg * 8 + row;
    float m = -INFINITY, total = 0;
    for (uint kb = 0; kb < qstart + BM; kb += BN) {
        for (uint i = tid; i < BN * 64; i += BM * 4) {
            ks[i] = half(K[base + kb * 64 + i]);
            vs[i] = half(V[base + kb * 64 + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[BN / 8];
        #pragma unroll
        for (uint c = 0; c < BN / 8; ++c)
            scores[c] = simdgroup_float8x8(0.0f);
        #pragma unroll
        for (uint d = 0; d < 8; ++d) {
            #pragma unroll
            for (uint c = 0; c < BN / 8; ++c) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, ks + c * 8 * 64 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[c], qa[d], kt, scores[c]);
            }
        }
        float mx = -INFINITY;
        #pragma unroll
        for (uint c = 0; c < BN / 8; ++c) {
            #pragma unroll
            for (uint j = 0; j < 2; ++j) {
                float s = scores[c].thread_elements()[j] * 0.125f;
                if (kb + c * 8 + col + j > qrow) s = -INFINITY;
                scores[c].thread_elements()[j] = s;
                mx = max(mx, s);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        float nm = max(m, mx);
        float correction = fast::exp(m - nm);
        float sum = 0;
        simdgroup_half8x8 probs[BN / 8];
        #pragma unroll
        for (uint c = 0; c < BN / 8; ++c) {
            #pragma unroll
            for (uint j = 0; j < 2; ++j) {
                float p = fast::exp(scores[c].thread_elements()[j] - nm);
                probs[c].thread_elements()[j] = half(p);
                sum += p;
            }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        total = total * correction + sum;
        m = nm;
        #pragma unroll
        for (uint d = 0; d < 8; ++d) {
            out[d].thread_elements()[0] *= correction;
            out[d].thread_elements()[1] *= correction;
            #pragma unroll
            for (uint c = 0; c < BN / 8; ++c) {
                simdgroup_half8x8 vt;
                simdgroup_load(vt, vs + c * 8 * 64 + d * 8, 64);
                simdgroup_multiply_accumulate(out[d], probs[c], vt, out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint d = 0; d < 8; ++d) {
        out[d].thread_elements()[0] /= total;
        out[d].thread_elements()[1] /= total;
        simdgroup_store(out[d], Y + base + (qstart + sg * 8) * 64 + d * 8, 64);
    }
}
'''
_LIB = torch.mps.compile_shader(_SOURCE.replace('__BM__', str(_BM)).replace('__BN__', str(_BN)))


def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64) or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = torch.empty_like(q)
    _LIB.attention(q, k, v, y, threads=(_BM * 4, 1024 // _BM, 64), group_size=(_BM * 4, 1, 1))
    return y
