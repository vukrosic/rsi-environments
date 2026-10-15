"""Fused causal attention for the fixed MPS workload."""
import torch

_SOURCE = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kv[64 * 64];
    const uint base = gid.y * 1024 * 64;
    const uint qb = gid.x * 64;
    const uint lane_col = (lane % 2) * 2 + ((lane / 8) % 2) * 4;
    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 out[8];
    for (uint d = 0; d < 8; ++d) {
        qm[d].thread_elements()[0] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col]);
        qm[d].thread_elements()[1] = half(Q[base + (qb + row) * 64 + d * 8 + lane_col + 1]);
        out[d] = simdgroup_float8x8(0.0f);
    }
    float mx = -INFINITY;
    float den = 0.0f;
    for (uint kb = 0; kb <= qb; kb += 64) {
        for (uint i = tid; i < 4096; i += 256)
            kv[i] = half(K[base + kb * 64 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 score[8];
        for (uint j = 0; j < 8; ++j) {
            score[j] = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(score[j], qm[d], km, score[j]);
            }
        }
        float tilemax = -INFINITY;
        for (uint j = 0; j < 8; ++j) {
            for (uint e = 0; e < 2; ++e) {
                uint col = lane_col + j * 8 + e;
                float z = kb + col <= qb + row ? score[j].thread_elements()[e] * 0.125f : -INFINITY;
                score[j].thread_elements()[e] = z;
                tilemax = max(tilemax, z);
            }
        }
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
        float newmx = max(mx, tilemax);
        float alpha = fast::exp(mx - newmx);
        float tilesum = 0.0f;
        simdgroup_half8x8 prob[8];
        for (uint j = 0; j < 8; ++j) {
            for (uint e = 0; e < 2; ++e) {
                float p = fast::exp(score[j].thread_elements()[e] - newmx);
                prob[j].thread_elements()[e] = half(p);
                tilesum += p;
            }
        }
        tilesum += simd_shuffle_xor(tilesum, 1);
        tilesum += simd_shuffle_xor(tilesum, 8);
        den = den * alpha + tilesum;
        mx = newmx;
        for (uint d = 0; d < 8; ++d) {
            out[d].thread_elements()[0] *= alpha;
            out[d].thread_elements()[1] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 4096; i += 256)
            kv[i] = half(V[base + kb * 64 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8; ++j) {
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
                simdgroup_multiply_accumulate(out[d], prob[j], vm, out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < 8; ++d) {
        out[d].thread_elements()[0] /= den;
        out[d].thread_elements()[1] /= den;
        simdgroup_store(out[d], O + base + (qb + sg * 8) * 64 + d * 8, 64);
    }
}
'''
_LIB = torch.mps.compile_shader(_SOURCE)

def kernel(q, k, v):
    out = torch.empty_like(q)
    _LIB.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
    return out
