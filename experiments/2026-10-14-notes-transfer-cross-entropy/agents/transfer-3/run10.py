"""Fused float32 Metal cross entropy for 8192 by 4096 logits."""
import torch

_source = r'''
#include <metal_stdlib>
using namespace metal;
kernel void rows(device const float4* x [[buffer(0)]],
                 device const long* y [[buffer(1)]],
                 device float* out [[buffer(2)]],
                 uint group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group;
    uint rt=tid;
    float4 v[8];
    float m = -INFINITY;
    v[0]=x[row*1024+rt+0];
    m=max(m,max(max(v[0].x,v[0].y),max(v[0].z,v[0].w)));
    v[1]=x[row*1024+rt+128];
    m=max(m,max(max(v[1].x,v[1].y),max(v[1].z,v[1].w)));
    v[2]=x[row*1024+rt+256];
    m=max(m,max(max(v[2].x,v[2].y),max(v[2].z,v[2].w)));
    v[3]=x[row*1024+rt+384];
    m=max(m,max(max(v[3].x,v[3].y),max(v[3].z,v[3].w)));
    v[4]=x[row*1024+rt+512];
    m=max(m,max(max(v[4].x,v[4].y),max(v[4].z,v[4].w)));
    v[5]=x[row*1024+rt+640];
    m=max(m,max(max(v[5].x,v[5].y),max(v[5].z,v[5].w)));
    v[6]=x[row*1024+rt+768];
    m=max(m,max(max(v[6].x,v[6].y),max(v[6].z,v[6].w)));
    v[7]=x[row*1024+rt+896];
    m=max(m,max(max(v[7].x,v[7].y),max(v[7].z,v[7].w)));
    threadgroup float shared[4];
    m = simd_max(m);
    if (lane==0) shared[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=simd_max(lane<4 ? shared[lane] : -INFINITY);
    float s=0;
    float4 e0=fast::exp(v[0]-m);
    s+=e0.x+e0.y+e0.z+e0.w;
    float4 e1=fast::exp(v[1]-m);
    s+=e1.x+e1.y+e1.z+e1.w;
    float4 e2=fast::exp(v[2]-m);
    s+=e2.x+e2.y+e2.z+e2.w;
    float4 e3=fast::exp(v[3]-m);
    s+=e3.x+e3.y+e3.z+e3.w;
    float4 e4=fast::exp(v[4]-m);
    s+=e4.x+e4.y+e4.z+e4.w;
    float4 e5=fast::exp(v[5]-m);
    s+=e5.x+e5.y+e5.z+e5.w;
    float4 e6=fast::exp(v[6]-m);
    s+=e6.x+e6.y+e6.z+e6.w;
    float4 e7=fast::exp(v[7]-m);
    s+=e7.x+e7.y+e7.z+e7.w;
    s=simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) shared[sg]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(rt==0) {
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
    losses = torch.empty((8192,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _lib.rows(logits, y, losses, threads=8192*128, group_size=128)
    _lib.finish(losses, out, threads=256, group_size=256)
    return out
