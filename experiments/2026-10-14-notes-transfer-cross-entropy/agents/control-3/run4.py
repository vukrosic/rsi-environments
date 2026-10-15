"""Fused row-wise cross entropy and GPU reduction."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void row_loss(device const float4* x [[buffer(0)]],
                     device const long* y [[buffer(1)]],
                     device float* loss [[buffer(2)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint row = group * 8 + simd;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < 32; ++i)
        acc += exp(x[row * 1024 + lane + i * 32]);
    float sum = simd_sum(acc.x + acc.y + acc.z + acc.w);
    float mx = 0.0f;
    if (!(sum > 1.0e-30f && isfinite(sum))) {
        float4 vmax = float4(-INFINITY);
        for (uint i = 0; i < 32; ++i)
            vmax = max(vmax, x[row * 1024 + lane + i * 32]);
        mx = simd_max(max(max(vmax.x, vmax.y), max(vmax.z, vmax.w)));
        acc = float4(0.0f);
        for (uint i = 0; i < 32; ++i)
            acc += exp(x[row * 1024 + lane + i * 32] - mx);
        sum = simd_sum(acc.x + acc.y + acc.z + acc.w);
    }
    if (lane == 0) {
        device const float* xf = reinterpret_cast<device const float*>(x);
        scratch[simd] = log(sum) + mx - xf[row * 4096 + uint(y[row])];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float value = 0.0f;
        for (uint i = 0; i < 8; ++i) value += scratch[i];
        loss[group] = value;
    }
}
kernel void finish(device const float* loss [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    float val = 0.0f;
    for (uint i = tid; i < 1024; i += 256) val += loss[i];
    val = simd_sum(val);
    if (lane == 0) scratch[simd] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        val = 0.0f;
        for (uint i = 0; i < 8; ++i) val += scratch[i];
        out[0] = val * (1.0f / 8192.0f);
    }
}
""")


def kernel(logits, y):
    loss = torch.empty((1024,), device=logits.device, dtype=logits.dtype)
    out = torch.empty((), device=logits.device, dtype=logits.dtype)
    _lib.row_loss(logits, y, loss, threads=1024 * 256, group_size=256)
    _lib.finish(loss, out, threads=256, group_size=256)
    return out
