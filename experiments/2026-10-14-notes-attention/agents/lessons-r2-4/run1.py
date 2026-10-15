"""Fused causal attention for the task's M4 inputs."""
import torch


_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kv[8192];
    threadgroup float scores[4096];
    uint base = tg.y * 1024 * 64;
    uint qb = tg.x * 64;
    for (uint i = tid; i < 4096; i += 256)
        kv[i] = half(Q[base + qb * 64 + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    for (uint d = 0; d < 8; ++d)
        simdgroup_load(qm[d], kv + sg * 512 + d * 8, 64);
    simdgroup_float8x8 acc[8];
    for (uint d = 0; d < 8; ++d)
        acc[d] = simdgroup_float8x8(0.0f);
    uint row = (lane / 16) * 4 + (lane % 8) / 2;
    uint col = (lane % 2) * 2 + (lane % 16 / 8) * 4;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint kb = 0; kb <= qb; kb += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 4096; i += 256) {
            kv[i] = half(K[base + kb * 64 + i]);
            kv[4096 + i] = half(V[base + kb * 64 + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sm[8];
        for (uint j = 0; j < 8; ++j)
            sm[j] = simdgroup_float8x8(0.0f);
        for (uint d = 0; d < 8; ++d) {
            for (uint j = 0; j < 8; ++j) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
            }
        }
        for (uint j = 0; j < 8; ++j)
            simdgroup_store(sm[j], scores + sg * 512 + j * 8, 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = sg * 8 + row;
        float local_max = -INFINITY;
        for (uint j = col; j < 64; j += 8) {
            float a = kb + j <= qb + r ? scores[r * 64 + j] * 0.125f : -INFINITY;
            float b = kb + j + 1 <= qb + r ? scores[r * 64 + j + 1] * 0.125f : -INFINITY;
            scores[r * 64 + j] = a;
            scores[r * 64 + j + 1] = b;
            local_max = max(local_max, max(a, b));
        }
        local_max = max(local_max, simd_shuffle_xor(local_max, 1));
        local_max = max(local_max, simd_shuffle_xor(local_max, 8));
        float new_max = max(running_max, local_max);
        float alpha = fast::exp(running_max - new_max);
        float local_sum = 0.0f;
        for (uint j = col; j < 64; j += 8) {
            float a = fast::exp(scores[r * 64 + j] - new_max);
            float b = fast::exp(scores[r * 64 + j + 1] - new_max);
            kv[r * 64 + j] = half(a);
            kv[r * 64 + j + 1] = half(b);
            local_sum += a + b;
        }
        local_sum += simd_shuffle_xor(local_sum, 1);
        local_sum += simd_shuffle_xor(local_sum, 8);
        running_sum = running_sum * alpha + local_sum;
        running_max = new_max;
        for (uint d = 0; d < 8; ++d) {
            acc[d].thread_elements()[0] *= alpha;
            acc[d].thread_elements()[1] *= alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8; ++j) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, kv + sg * 512 + j * 8, 64);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
                simdgroup_multiply_accumulate(acc[d], pm, vm, acc[d]);
            }
        }
    }
    for (uint d = 0; d < 8; ++d) {
        acc[d].thread_elements()[0] /= running_sum;
        acc[d].thread_elements()[1] /= running_sum;
        simdgroup_store(acc[d], O + base + (qb + sg * 8) * 64 + d * 8, 64);
    }
}
""")


def kernel(q, k, v):
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out, threads=(16 * 256, 64, 1), group_size=(256, 1, 1))
    return out
