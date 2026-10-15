import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void softmax_causal(device half* scores [[buffer(0)]],
                           uint tid [[thread_position_in_grid]]) {
    uint row=tid/32;
    uint lane=tid%32;
    uint pos=row%1024;
    device half2* data=reinterpret_cast<device half2*>(scores+row*1024);
    float2 vals[16];
    float mx=-INFINITY;
    for(uint j=0;j<16;++j) {
        uint c=lane*2+j*64;
        vals[j]=float2(data[lane+j*32])*0.125f;
        if(c>pos) vals[j].x=-INFINITY;
        if(c+1>pos) vals[j].y=-INFINITY;
        mx=max(mx,max(vals[j].x,vals[j].y));
    }
    mx=simd_max(mx);
    float sum=0;
    for(uint j=0;j<16;++j) { vals[j]=exp(vals[j]-mx); sum+=vals[j].x+vals[j].y; }
    sum=simd_sum(sum);
    for(uint j=0;j<16;++j) data[lane+j*32]=half2(vals[j]/sum);
}

""")

def kernel(q,k,v):
    qh,kh,vh=q.half(),k.half(),v.half()
    scores=qh @ kh.transpose(-1,-2)
    _lib.softmax_causal(scores,threads=(65536*32,),group_size=(128,))
    return (scores @ vh).float()
