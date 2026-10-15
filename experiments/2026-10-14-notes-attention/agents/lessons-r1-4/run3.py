"""Causal attention with half precision matrix products and fused softmax."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void causal_softmax(device const half* scores [[buffer(0)]],
                           device half* probs [[buffer(1)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint pos = row % 1024;
    uint base = row * 1024;
    float x[4];
    float mx = -INFINITY;
    for (uint i=0; i<4; ++i) {
        uint col = tid + i*256;
        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
        mx = max(mx,x[i]);
    }
    mx = simd_max(mx);
    if (lane==0) scratch[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = tid<8 ? scratch[tid] : -INFINITY;
    mx = simd_max(mx);
    if (tid==0) scratch[0]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=scratch[0];
    float sum=0;
    for (uint i=0; i<4; ++i) { x[i]=fast::exp(x[i]-mx); sum+=x[i]; }
    sum=simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=tid<8?scratch[tid]:0;
    sum=simd_sum(sum);
    if(tid==0) scratch[0]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.0f/scratch[0];
    for(uint i=0;i<4;++i) probs[base+tid+i*256]=half(x[i]*inv);
}
""")

def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    scores = qh @ kh.transpose(-1,-2)
    probs = torch.empty_like(scores)
    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*256), group_size=256)
    return (probs @ vh).float()
