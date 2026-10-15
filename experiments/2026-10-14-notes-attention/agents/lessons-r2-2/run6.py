"""Fused causal attention for the fixed MPS workload."""
import torch

_BQ = 64
_BK = 32

_SOURCE = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define BQ __BQ__
#define BK __BK__

kernel void attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kv[(BQ > BK ? BQ : BK) * 64];
    threadgroup float scores[BQ * BK];
    threadgroup half probs[BQ * BK];
    const uint base = gid.y * 1024 * 64;
    const uint qb = gid.x * BQ;
    for (uint i = tid; i < BQ * 64; i += BQ * 4)
        kv[i] = half(Q[base + qb * 64 + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 out[8];
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(qm[d], kv + sg * 8 * 64 + d * 8, 64);
        out[d] = simdgroup_float8x8(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    float den = 0.0f;
    const uint lane_col = lane % 2 + ((lane / 8) % 2) * 2;
    const uint row = sg * 8 + (lane % 8) / 2 + (lane / 16) * 4;
    for (uint kb = 0; kb < qb + BQ; kb += BK) {
        for (uint i = tid; i < BK * 64; i += BQ * 4)
            kv[i] = half(K[base + kb * 64 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < BK / 8; ++j) {
            simdgroup_float8x8 acc(0.0f);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kv + j * 8 * 64 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(acc, qm[d], km, acc);
            }
            simdgroup_store(acc, scores + sg * 8 * BK + j * 8, BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vals[BK / 4];
        float tilemax = -INFINITY;
        for (uint j = 0; j < BK / 4; ++j) {
            uint col = lane_col + j * 4;
            float z = kb + col <= qb + row ? scores[row * BK + col] * 0.125f : -INFINITY;
            vals[j] = z;
            tilemax = max(tilemax, z);
        }
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
        float newmx = max(mx, tilemax);
        float alpha = fast::exp(mx - newmx);
        float tilesum = 0.0f;
        for (uint j = 0; j < BK / 4; ++j) {
            float p = fast::exp(vals[j] - newmx);
            probs[row * BK + lane_col + j * 4] = half(p);
            tilesum += p;
        }
        tilesum += simd_shuffle_xor(tilesum, 1);
        tilesum += simd_shuffle_xor(tilesum, 8);
        den = den * alpha + tilesum;
        mx = newmx;
        for (uint d = 0; d < 8; ++d) {
            out[d].thread_elements()[0] *= alpha;
            out[d].thread_elements()[1] *= alpha;
        }
        for (uint i = tid; i < BK * 64; i += BQ * 4)
            kv[i] = half(V[base + kb * 64 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < BK / 8; ++j) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, probs + sg * 8 * BK + j * 8, BK);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, kv + j * 8 * 64 + d * 8, 64);
                simdgroup_multiply_accumulate(out[d], pm, vm, out[d]);
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
_LIB = torch.mps.compile_shader(_SOURCE.replace("__BQ__", str(_BQ)).replace("__BK__", str(_BK)))

def kernel(q, k, v):
    out = torch.empty_like(q)
    _LIB.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_BQ * 4, 1, 1))
    return out
