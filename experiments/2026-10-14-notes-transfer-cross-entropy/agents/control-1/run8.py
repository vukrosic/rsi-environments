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
    threadgroup float maxima[16];
    threadgroup float sums[16];
    float values[8];
    float vmax = -INFINITY;
    for (uint k = 0; k < 8; ++k) {
        values[k] = x[row * 4096 + tid + k * 512];
        vmax = max(vmax, values[k]);
    }
    vmax = simd_max(vmax);
    if (lane == 0) maxima[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(lane < 16 ? maxima[lane] : -INFINITY);
    float total = 0.0f;
    for (uint k = 0; k < 8; ++k)
        total += fast::exp(values[k] - vmax);
    total = simd_sum(total);
    if (lane == 0) sums[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(lane < 16 ? sums[lane] : 0.0f);
    if (tid == 0)
        losses[row] = fast::log(total) + vmax - x[row * cols + labels[row]];
}

kernel void mean_loss(
    device const float* losses,
    device float* result,
    constant uint& rows,
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    float total = 0.0f;
    for (uint row = tid; row < rows; row += 256)
        total += losses[row];
    total = simd_sum(total);
    if (lane == 0) scratch[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
    if (tid == 0) result[0] = total / float(rows);
}
""")


def kernel(logits, y):
    rows, cols = logits.shape
    losses = torch.empty(rows, dtype=logits.dtype, device=logits.device)
    result = torch.empty((), dtype=logits.dtype, device=logits.device)
    _lib.cross_entropy(logits, y, losses, cols,
                       threads=rows * 512, group_size=512)
    _lib.mean_loss(losses, result, rows, threads=256, group_size=256)
    return result
