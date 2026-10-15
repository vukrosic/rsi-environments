import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void rows(device const float4* x,
                 device const uint2* labels,
                 device float* losses,
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    float4 accum = float4(0);
    for (uint k = 0; k < 4; ++k) {
        accum += exp(x[row * 1024 + tid + k * 256]);
    }
    float s = simd_sum(accum.x + accum.y + accum.z + accum.w);
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) {
            device const float* scalar = reinterpret_cast<device const float*>(x);
            losses[row] = log(s) - scalar[row * 4096 + labels[row].x];
        }
    }
}

kernel void finish(device const float* losses, device float* result,
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    float sum = 0;
    for (uint i = tid; i < 8192; i += 128) sum += losses[i];
    sum = simd_sum(sum);
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        sum = 0;
        for (uint i = 0; i < 4; ++i) sum += scratch[i];
        result[0] = sum / 8192.0f;
    }
}
""")


def kernel(logits, y):
    losses = torch.empty((8192,), device=logits.device, dtype=torch.float32)
    result = torch.empty((), device=logits.device, dtype=torch.float32)
    _lib.rows(logits, y, losses, threads=8192 * 256, group_size=256)
    _lib.finish(losses, result, threads=128, group_size=128)
    return result
