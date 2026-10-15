import torch

_source = r'''
#include <metal_stdlib>
using namespace metal;
kernel void rows(device const float4* x [[buffer(0)]],
                 device const long* y [[buffer(1)]],
                 device float* out [[buffer(2)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    half4 v[8];
    float m = -INFINITY;
    #pragma unroll
    for (uint i=0; i<8; ++i) {
        v[i] = half4(x[row*1024 + tid + i*128]);
        m = max(m, max(max(v[i].x,v[i].y),max(v[i].z,v[i].w)));
    }
    threadgroup float shared[4];
    m = simd_max(m);
    if (lane==0) shared[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
    float s=0;
    #pragma unroll
    for (uint i=0;i<8;++i) {
        float4 e=fast::exp(float4(v[i])-m);
        s+=e.x+e.y+e.z+e.w;
    }
    s=simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) shared[sg]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        s=0;
        for(uint i=0;i<4;++i) s+=shared[i];
        out[row]=log(s)+m-((device const float*)x)[row*4096+y[row]];
    }
}
kernel void finish(device const float* rows [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    float s=0;
    for(uint i=tid;i<8192;i+=256) s+=rows[i];
    s=simd_sum(s);
    threadgroup float shared[8];
    if(lane==0) shared[sg]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        s=0;
        for(uint i=0;i<8;++i) s+=shared[i];
        out[0]=s/8192.0f;
    }
}
'''
_lib = torch.mps.compile_shader(_source)

def kernel(logits, y):
    losses = torch.empty((8192,), device=logits.device)
    out = torch.empty((), device=logits.device)
    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
    _lib.finish(losses, out, threads=256, group_size=256)
    return out
