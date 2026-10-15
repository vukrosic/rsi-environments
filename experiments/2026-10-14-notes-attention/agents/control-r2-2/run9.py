import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void softmax_causal(device half* scores [[buffer(0)]],
                           constant uint& n [[buffer(1)]],
                           constant uint& offset [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint pos = row % 128 + offset;
    uint base = row * n;
    float vals[4];
    float mx = -INFINITY;
    for (uint j=0; j<4; ++j) {
        uint c = tid + j*256;
        vals[j] = c < n && c <= pos ? float(scores[base+c])*0.125f : -INFINITY;
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
    for(uint j=0;j<4;++j) if(tid+j*256<n) scores[base+tid+j*256]=half(vals[j]/sum);
}
""")

def kernel(q,k,v):
    qh,kh,vh=q.half(),k.half(),v.half()
    chunks=[]
    for offset in range(0,1024,128):
        n=offset+128
        scores=qh[...,offset:n,:] @ kh[...,:n,:].transpose(-1,-2)
        _lib.softmax_causal(scores,n,offset,threads=(8192*256,),group_size=(256,))
        chunks.append(scores @ vh[...,:n,:])
    return torch.cat(chunks,dim=-2).float()
