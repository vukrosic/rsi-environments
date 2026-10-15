import torch
import torch.nn.functional as F

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;

template<uint N>
inline void softmax_impl(device half* scores, uint offset, uint group, uint sg, uint lane) {
    uint row=group*4+sg;
    uint base=row*N;
    uint causal=offset+row%256;
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
DEFINE(256)
DEFINE(512)
DEFINE(768)
DEFINE(1024)
''')
_softmax = (_lib.sm256, _lib.sm512, _lib.sm768, _lib.sm1024)


def kernel(q, k, v):
    """Causal attention with half-precision GEMMs and float32 softmax reductions."""
    if q.shape != (8, 8, 1024, 64):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Restrict each query block to the key prefix it can attend to.
    qh, kh, vh = (x.half().view(64, 1024, 64) for x in (q, k, v))
    out = torch.empty_like(qh)
    for i, start in enumerate(range(0, 1024, 256)):
        end = start + 256
        scores = torch.bmm(qh[:, start:end], kh[:, :end].transpose(1, 2))
        _softmax[i](scores, start, threads=(64*256*32, 1, 1), group_size=(128, 1, 1))
        torch.bmm(scores, vh[:, :end], out=out[:, start:end])
    return out.view_as(q)
