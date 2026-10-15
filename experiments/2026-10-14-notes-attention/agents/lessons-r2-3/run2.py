import torch
import torch.nn.functional as F


_BQ = 64
_BK = 64
_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint BQ = 64;
constant uint BK = 64;
kernel void attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half stage[(BQ > BK ? BQ : BK) * 64];
    const uint base = group.y * 1024 * 64;
    const uint query = group.x * BQ;
    for (uint i = tid; i < BQ * 64; i += 256)
        stage[i] = half(q[base + query * 64 + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 acc[8];
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(qm[d], stage + sg * 8 * 64 + d * 8, 64);
        acc[d] = simdgroup_float8x8(0.0f);
    }
    float maxima = -INFINITY, denom = 0.0f;
    const uint matrix_row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
    const uint matrix_col = ((lane & 8) >> 1) + ((lane & 1) << 1);
    const uint row = query + sg * 8 + matrix_row;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = 0; key < query + BQ; key += BK) {
        for (uint i = tid; i < BK * 64; i += 256)
            stage[i] = half(k[base + key * 64 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[BK / 8];
        for (uint c = 0; c < BK / 8; ++c) {
            scores[c] = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 km;
                simdgroup_load(km, stage + c * 8 * 64 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[c], qm[d], km, scores[c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK * 64; i += 256)
            stage[i] = half(v[base + key * 64 + i]);
        float mx = -INFINITY;
        for (uint c = 0; c < BK / 8; ++c) {
            for (uint i = 0; i < 2; ++i) {
                float s = scores[c].thread_elements()[i] * 0.125f;
                s = key + c * 8 + matrix_col + i <= row ? s : -INFINITY;
                scores[c].thread_elements()[i] = s;
                mx = max(mx, s);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        mx = max(mx, maxima);
        float alpha = fast::exp(maxima - mx);
        simdgroup_half8x8 pm[BK / 8];
        float sum = 0.0f;
        for (uint c = 0; c < BK / 8; ++c) {
            for (uint i = 0; i < 2; ++i) {
                float p = fast::exp(scores[c].thread_elements()[i] - mx);
                pm[c].thread_elements()[i] = half(p);
                sum += p;
            }
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        denom = denom * alpha + sum;
        maxima = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < 8; ++d) {
            acc[d].thread_elements()[0] *= alpha;
            acc[d].thread_elements()[1] *= alpha;
            for (uint c = 0; c < BK / 8; ++c) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, stage + c * 8 * 64 + d * 8, 64);
                simdgroup_multiply_accumulate(acc[d], pm[c], vm, acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < 8; ++d) {
        acc[d].thread_elements()[0] /= denom;
        acc[d].thread_elements()[1] /= denom;
        simdgroup_store(acc[d], out + base + (query + sg * 8) * 64 + d * 8, 64);
    }
}
"""
_LIB = torch.mps.compile_shader(_SOURCE)


def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64) or not all(x.is_contiguous() for x in (q, k, v)):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = torch.empty_like(q)
    _LIB.attention(q, k, v, out, threads=(1024 // _BQ * 256, 64, 1), group_size=(256, 1, 1))
    return out
