"""Fused stable cross entropy on MPS."""
import torch

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
kernel void row_loss(device const float4* x [[buffer(0)]],
                     device const long* y [[buffer(1)]],
                     device float* losses [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    threadgroup float sums[8];
    float4 v[4];
    float m = -INFINITY;
    #pragma unroll
    for (uint i=0; i<4; ++i) {
        v[i] = x[row*1024 + tid + i*256];
        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
    }
    m = simd_max(m);
    if (lane == 0) scratch[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
    float s = 0;
    #pragma unroll
    for (uint i=0; i<4; ++i) {
        float4 e = fast::exp2((v[i]-m)*1.4426950408889634f);
        s += e.x+e.y+e.z+e.w;
    }
    s = simd_sum(s);
    if (lane == 0) sums[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane==0) losses[row] = log(s)+m-((device const float*)x)[row*4096+uint(y[row])];
    }
}
kernel void mean_loss(device const float* losses [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    float s = 0;
    for(uint i=tid; i<8192; i+=256) s += losses[i];
    s = simd_sum(s);
    if(lane==0) scratch[sg]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = simd_sum(lane<8 ? scratch[lane] : 0.0f);
    if(tid==0) out[0]=s/8192.0f;
}
'''
_LIB = torch.mps.compile_shader(_SOURCE)

def kernel(logits, y):
    losses = torch.empty((8192,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _LIB.row_loss(logits, y, losses, threads=(8192*256,1,1), group_size=(256,1,1))
    _LIB.mean_loss(losses, out, threads=256, group_size=256)
    return out
