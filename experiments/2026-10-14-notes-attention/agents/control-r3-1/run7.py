import torch
import torch.nn.functional as F

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void softmax_causal(device half* scores [[buffer(0)]],
                          constant uint& n [[buffer(1)]],
                          constant uint& offset [[buffer(2)]],
                          uint3 gid [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shared[8];
    uint row=gid.x;
    uint base=row*n;
    uint causal=offset+row%256;
    float values[4];
    float m=-INFINITY;
    #pragma unroll
    for(uint i=0;i<4;i++) {
        uint col=tid+256*i;
        float x=col<n && col<=causal ? float(scores[base+col])*0.125f : -INFINITY;
        values[i]=x;
        m=max(m,x);
    }
    m=simd_max(m);
    if(lane==0) shared[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=-INFINITY;
    for(uint i=0;i<8;i++) m=max(m,shared[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0.0f;
    #pragma unroll
    for(uint i=0;i<4;i++) {
        values[i]=fast::exp(values[i]-m);
        sum+=values[i];
    }
    sum=simd_sum(sum);
    if(lane==0) shared[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0.0f;
    for(uint i=0;i<8;i++) sum+=shared[i];
    #pragma unroll
    for(uint i=0;i<4;i++) {
        uint col=tid+256*i;
        if(col<n) scores[base+col]=half(values[i]/sum);
    }
}
''')


def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    qh, kh, vh = (x.half().view(64, 1024, 64) for x in (q, k, v))
    out = torch.empty_like(qh)
    for start in range(0, 1024, 256):
        end = start + 256
        scores = torch.bmm(qh[:, start:end], kh[:, :end].transpose(1, 2))
        _lib.softmax_causal(scores, end, start, threads=(64*256*256, 1, 1), group_size=(256, 1, 1))
        torch.bmm(scores, vh[:, :end], out=out[:, start:end])
    return out.view_as(q).float()
