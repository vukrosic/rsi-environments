"""Fused row-wise cross entropy and GPU reduction."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void row_loss(device const float* x [[buffer(0)]],
                     device const long* y [[buffer(1)]],
                     device float* loss [[buffer(2)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint local_row = tid / 128;
    uint row = group * 2 + local_row;
    uint rtid = tid % 128;
    float sum = 0.0f;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    float4 acc = float4(0.0f);
    #pragma unroll
    for (uint i = 0; i < 8; ++i)
        acc += fast::exp(x4[row * 1024 + rtid + i * 128]);
    sum = acc.x + acc.y + acc.z + acc.w;
    sum = simd_sum(sum);
    if (lane == 0) scratch[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum0 = simd_sum(lane < 4 ? scratch[lane] : 0.0f);
    float sum1 = simd_sum(lane < 4 ? scratch[lane + 4] : 0.0f);
    sum = local_row == 0 ? sum0 : sum1;
    if (sum0 > 1.0e-30f && isfinite(sum0) && sum1 > 1.0e-30f && isfinite(sum1)) {
        if (rtid == 0) loss[row] = log(sum) - x[row * 4096 + uint(y[row])];
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx = -INFINITY;
        for (uint i = 0; i < 32; ++i)
            mx = max(mx, x[row * 4096 + rtid + i * 128]);
        mx = simd_max(mx);
        if (lane == 0) scratch[simd] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mx = -INFINITY;
        for (uint i = 0; i < 4; ++i) mx = max(mx, scratch[local_row * 4 + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = 0.0f;
        for (uint i = 0; i < 32; ++i)
            sum += fast::exp(x[row * 4096 + rtid + i * 128] - mx);
        sum = simd_sum(sum);
        if (lane == 0) scratch[simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rtid == 0) {
            sum = 0.0f;
            for (uint i = 0; i < 4; ++i) sum += scratch[local_row * 4 + i];
            loss[row] = log(sum) + mx - x[row * 4096 + uint(y[row])];
        }
    }
}
kernel void finish(device const float* loss [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[4];
    float val = 0.0f;
    for (uint i = tid; i < 8192; i += 128) val += loss[i];
    val = simd_sum(val);
    if (lane == 0) scratch[simd] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        val = 0.0f;
        for (uint i = 0; i < 4; ++i) val += scratch[i];
        out[0] = val * (1.0f / 8192.0f);
    }
}
""")


def kernel(logits, y):
    loss = torch.empty((8192,), device=logits.device, dtype=logits.dtype)
    out = torch.empty((), device=logits.device, dtype=logits.dtype)
    _lib.row_loss(logits, y, loss, threads=8192 * 128, group_size=256)
    _lib.finish(loss, out, threads=128, group_size=128)
    return out
