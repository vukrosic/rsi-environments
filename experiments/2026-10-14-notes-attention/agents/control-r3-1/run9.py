import torch
import torch.nn.functional as F

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;

template<uint N>
inline void softmax_impl(device half* scores, uint offset, uint group, uint sg, uint lane) {
    uint row=group*4+sg;
    uint base=row*N;
    uint causal=offset+row%128;
    float values[N/32];
    float m=-INFINITY;
    #pragma unroll
    for(uint i=0;i<N/32;i++) {
        uint col=lane+32*i;
        float x=col<=causal ? float(scores[base+col])*0.125f : -INFINITY;
        values[i]=x;
        m=max(m,x);
    }
    m=simd_max(m);
    float sum=0.0f;
    #pragma unroll
    for(uint i=0;i<N/32;i++) {
        values[i]=fast::exp(values[i]-m);
        sum+=values[i];
    }
    sum=simd_sum(sum);
    #pragma unroll
    for(uint i=0;i<N/32;i++) {
        scores[base+lane+32*i]=half(values[i]/sum);
    }
}
#define DEFINE(N) \
kernel void sm##N(device half* scores [[buffer(0)]], \
                 constant uint& offset [[buffer(1)]], \
                 uint group [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]]) { \
    softmax_impl<N>(scores,offset,group,sg,lane); \
}
DEFINE(128)
DEFINE(256)
DEFINE(384)
DEFINE(512)
DEFINE(640)
DEFINE(768)
DEFINE(896)
DEFINE(1024)
''')
_softmax = tuple(getattr(_lib, f'sm{n}') for n in range(128, 1025, 128))


def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    qh, kh, vh = (x.half().view(64, 1024, 64) for x in (q, k, v))
    out = torch.empty_like(qh)
    for i, start in enumerate(range(0, 1024, 128)):
        end = start + 128
        scores = torch.bmm(qh[:, start:end], kh[:, :end].transpose(1, 2))
        _softmax[i](scores, start, threads=(64*128*32, 1, 1), group_size=(128, 1, 1))
        torch.bmm(scores, vh[:, :end], out=out[:, start:end])
    return out.view_as(q).float()
