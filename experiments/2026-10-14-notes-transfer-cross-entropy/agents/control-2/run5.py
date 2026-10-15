"""Fused cross-entropy reduction for MPS."""
import torch

_shader = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;

kernel void row_loss(
    device const float4 *x4 [[buffer(0)]],
    device const long *y [[buffer(1)]],
    device float *loss [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 4 + simd;
    device const float *x = reinterpret_cast<device const float *>(x4);
    float s = 0.0f;
    float m = -INFINITY;
    for (uint i = 0; i < 32; ++i) {
        float4 v = x4[row * 1024 + lane + i * 32];
        m = max(m, max(max(v.x, v.y), max(v.z, v.w)));
        float4 e = fast::exp(v);
        s += e.x + e.y + e.z + e.w;
    }
    s = simd_sum(s);
    m = simd_max(m);
    float lse;
    if (isfinite(s) && s > 1.0e-30f) {
        lse = fast::log(s);
    } else {
        s = 0.0f;
        for (uint i = 0; i < 32; ++i) {
            float4 e = fast::exp(x4[row * 1024 + lane + i * 32] - m);
            s += e.x + e.y + e.z + e.w;
        }
        lse = m + fast::log(simd_sum(s));
    }
    if (lane == 0) loss[row] = lse - x[row * 4096 + y[row]];
}

kernel void mean_loss(
    device const float *loss [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    float s = 0.0f;
    for (uint i = tid; i < 8192; i += 256) s += loss[i];
    s = simd_sum(s);
    if (lane == 0) scratch[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(lane < 8 ? scratch[lane] : 0.0f);
    if (tid == 0) out[0] = s / 8192.0f;
}
""")


def kernel(logits, y):
    loss = torch.empty((8192,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _shader.row_loss(logits, y, loss, threads=8192 * 32, group_size=128)
    _shader.mean_loss(loss, out, threads=256, group_size=256)
    return out
