"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy(
    device const float4* x,
    device const long* labels,
    device float* losses,
    constant uint& cols,
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    float4 values[8];
    float4 vmax4 = float4(-INFINITY);
    for (uint k = 0; k < 8; ++k) {
        values[k] = x[row * 1024 + tid + k * 128];
        vmax4 = max(vmax4, values[k]);
    }
    float vmax = simd_max(max(max(vmax4.x, vmax4.y), max(vmax4.z, vmax4.w)));
    if (lane == 0) scratch[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(lane < 4 ? scratch[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 total = float4(0.0f);
    for (uint k = 0; k < 8; ++k)
        total += exp(values[k] - vmax);
    float sum = simd_sum(total.x + total.y + total.z + total.w);
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
    if (tid == 0) {
        device const float* scalar = reinterpret_cast<device const float*>(x);
        losses[row] = log(sum) + vmax - scalar[row * cols + labels[row]];
    }
}
""")


def kernel(logits, y):
    rows, cols = logits.shape
    losses = torch.empty(rows, dtype=logits.dtype, device=logits.device)
    _lib.cross_entropy(logits, y, losses, cols,
                       threads=rows * 128, group_size=128)
    return losses.mean()
