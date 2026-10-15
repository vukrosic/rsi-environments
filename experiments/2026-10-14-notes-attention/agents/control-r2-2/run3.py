import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void softmax_causal(device half* scores [[buffer(0)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint pos = row % 1024;
    uint base = row * 1024;
    float vals[4];
    float mx = -INFINITY;
    for (uint j=0; j<4; ++j) {
        uint c = tid + j*256;
        vals[j] = c <= pos ? float(scores[base+c])*0.125f : -INFINITY;
        mx = max(mx, vals[j]);
    }
    mx = simd_max(mx);
    if(lane == 0) scratch[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = -INFINITY;
    for(uint j=0;j<8;++j) mx=max(mx,scratch[j]);
    float sum=0;
    for(uint j=0;j<4;++j) { vals[j]=exp(vals[j]-mx); sum+=vals[j]; }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane == 0) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0;
    for(uint j=0;j<8;++j) sum+=scratch[j];
    for(uint j=0;j<4;++j) scores[base+tid+j*256]=half(vals[j]/sum);
}
""")

def kernel(q,k,v):
    qh,kh,vh=q.half(),k.half(),v.half()
    scores=qh @ kh.transpose(-1,-2)
    _lib.softmax_causal(scores, threads=(65536*256,), group_size=(256,))
    return (scores @ vh).float()
