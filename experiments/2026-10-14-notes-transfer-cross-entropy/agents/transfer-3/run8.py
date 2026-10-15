import torch

_source = r'''
#include <metal_stdlib>
using namespace metal;
kernel void rows(device const float4* x [[buffer(0)]],
                 device const long* y [[buffer(1)]],
                 device float* out [[buffer(2)]],
                 uint group [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group*8+sg;
    float4 acc=0;
    #pragma unroll
    for(uint i=0;i<32;++i) acc+=fast::exp(x[row*1024+lane+i*32]);
    float s=simd_sum(acc.x+acc.y+acc.z+acc.w);
    if(lane==0) out[row]=log(s)-((device const float*)x)[row*4096+y[row]];
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
    _lib.rows(logits, y, losses, threads=8192*32, group_size=256)
    _lib.finish(losses, out, threads=256, group_size=256)
    return out
