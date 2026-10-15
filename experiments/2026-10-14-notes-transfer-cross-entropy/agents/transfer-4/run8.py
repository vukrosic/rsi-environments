"""Fused stable cross entropy on Metal."""
import torch

_SHADER = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void row_loss(
    device const float* logits [[buffer(0)]],
    device const long* labels [[buffer(1)]],
    device float* losses [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 2 + sg;
    float4 v[32];
    float mx = -INFINITY;
    #pragma unroll
    for (uint i=0; i<32; ++i) {
        v[i] = ((device const float4*)logits)[row*1024 + lane + i*32];
        mx = max(mx, max(max(v[i].x, v[i].y), max(v[i].z, v[i].w)));
    }
    mx = simd_max(mx);
    float s = 0.0f;
    #pragma unroll
    for (uint i=0; i<32; ++i) {
        float4 e = exp(v[i]-mx);
        s += e.x + e.y + e.z + e.w;
    }
    s = simd_sum(s);
    if (lane == 0) losses[row] = mx + log(s) - logits[row*4096 + labels[row]];
}
kernel void mean_loss(
    device const float* losses [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float s = 0.0f;
    for (uint i=tid; i<8192; i+=256) s += losses[i];
    s = simd_sum(s);
    if (lane == 0) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (tid == 0) out[0] = s / 8192.0f;
    }
}
""")


def kernel(logits, y):
    losses = torch.empty((8192,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _SHADER.row_loss(logits, y, losses, threads=8192*32, group_size=64)
    _SHADER.mean_loss(losses, out, threads=256, group_size=256)
    return out
