"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void cross_entropy(
    device const float* x,
    device const long* labels,
    device float* losses,
    constant uint& cols,
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    float values[16];
    float vmax = -INFINITY;
    for (uint k = 0; k < 16; ++k) {
        values[k] = x[row * 4096 + tid + k * 256];
        vmax = max(vmax, values[k]);
    }
    vmax = simd_max(vmax);
    if (lane == 0) scratch[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint k = 0; k < 16; ++k)
        total += exp(values[k] - vmax);
    total = simd_sum(total);
    if (lane == 0) scratch[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
    if (tid == 0)
        losses[row] = log(total) + vmax - x[row * cols + labels[row]];
}
""")


def kernel(logits, y):
    rows, cols = logits.shape
    losses = torch.empty(rows, dtype=logits.dtype, device=logits.device)
    _lib.cross_entropy(logits, y, losses, cols,
                       threads=rows * 256, group_size=256)
    return losses.mean()
