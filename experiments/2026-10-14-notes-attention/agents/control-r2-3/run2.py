import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void causal_softmax(device const half* s, device half* p,
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint last = row % 1024;
    float x[4];
    float mx = -INFINITY;
    for (uint i=0;i<4;i++) {
        uint col=tid+i*256;
        x[i] = col<=last ? float(s[row*1024+col])*0.125f : -INFINITY;
        mx=max(mx,x[i]);
    }
    mx=simd_max(mx);
    if(lane==0) scratch[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=simd_max(lane<8?scratch[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint i=0;i<4;i++){x[i]=fast::exp(x[i]-mx);sum+=x[i];}
    sum=simd_sum(sum);
    if(lane==0) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<8?scratch[lane]:0.f);
    for(uint i=0;i<4;i++) p[row*1024+tid+i*256]=half(x[i]/sum);
}
''')


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    scores = qh @ kh.transpose(-1, -2)
    p = torch.empty_like(scores)
    _lib.causal_softmax(scores, p, threads=(q.numel() // 64 * 256), group_size=256)
    return (p @ vh).float()
