"""Fused causal attention for the task's M4 inputs."""
import torch


_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(
    device const float4* Q [[buffer(0)]],
    device const float4* K [[buffer(1)]],
    device const float4* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // Stage queries once, then reuse the storage for key and value tiles.
    threadgroup half kv[8192];
    threadgroup half4* kv4 = (threadgroup half4*)kv;
    uint base = tg.y * 1024 * 64;
    uint qb = tg.x * 64;
    #pragma clang loop unroll(full)
    for (uint step = 0; step < 4; ++step) {
        uint i = tid + step * 256;
        kv4[i] = half4(Q[base / 4 + qb * 16 + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 8; ++d)
        simdgroup_load(qm[d], kv + sg * 512 + d * 8, 64);
    simdgroup_float8x8 acc[8];
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 8; ++d)
        acc[d] = simdgroup_float8x8(0.0f);
    // M4 matrix fragments assign two neighboring columns to each lane.
    uint row = (lane / 16) * 4 + (lane % 8) / 2;
    uint col = (lane % 2) * 2 + (lane % 16 / 8) * 4;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint kb = 0; kb <= qb; kb += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint step = 0; step < 4; ++step) {
            uint i = tid + step * 256;
            kv4[i] = half4(K[base / 4 + kb * 16 + i]);
            kv4[1024 + i] = half4(V[base / 4 + kb * 16 + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sm[8];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j)
            sm[j] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (uint d = 0; d < 8; ++d) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kv + j * 512 + d * 8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
            }
        }
        uint r = sg * 8 + row;
        if (kb == qb) {
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                if (j * 8 + col > r)
                    sm[j].thread_elements()[0] = -INFINITY;
                if (j * 8 + col + 1 > r)
                    sm[j].thread_elements()[1] = -INFINITY;
            }
        }
        float local_max = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            float a = sm[j].thread_elements()[0] * 0.125f;
            float b = sm[j].thread_elements()[1] * 0.125f;
            sm[j].thread_elements()[0] = a;
            sm[j].thread_elements()[1] = b;
            local_max = max(local_max, max(a, b));
        }
        local_max = max(local_max, simd_shuffle_xor(local_max, 1));
        local_max = max(local_max, simd_shuffle_xor(local_max, 8));
        float new_max = max(running_max, local_max);
        float alpha = fast::exp(running_max - new_max);
        float local_sum = 0.0f;
        simdgroup_half8x8 pm[8];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            float a = fast::exp(sm[j].thread_elements()[0] - new_max);
            float b = fast::exp(sm[j].thread_elements()[1] - new_max);
            pm[j].thread_elements()[0] = half(a);
            pm[j].thread_elements()[1] = half(b);
            local_sum += a + b;
        }
        local_sum += simd_shuffle_xor(local_sum, 1);
        local_sum += simd_shuffle_xor(local_sum, 8);
        running_sum = running_sum * alpha + local_sum;
        running_max = new_max;
        #pragma clang loop unroll(full)
        for (uint d = 0; d < 8; ++d) {
            acc[d].thread_elements()[0] *= alpha;
            acc[d].thread_elements()[1] *= alpha;
        }
        // Round only a sixteen-key partial sum; retain running output in float.
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; j += 2) {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_half8x8 vm, part(half(0));
                simdgroup_load(vm, kv + 4096 + j * 512 + d * 8, 64);
                simdgroup_multiply_accumulate(part, pm[j], vm, part);
                simdgroup_load(vm, kv + 4096 + (j + 1) * 512 + d * 8, 64);
                simdgroup_multiply_accumulate(part, pm[j + 1], vm, part);
                acc[d].thread_elements()[0] += float(part.thread_elements()[0]);
                acc[d].thread_elements()[1] += float(part.thread_elements()[1]);
            }
        }
    }
    #pragma clang loop unroll(full)
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
