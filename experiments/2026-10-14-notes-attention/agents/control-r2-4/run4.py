import torch
import torch.nn.functional as F

_src = r'''
#include <metal_stdlib>
using namespace metal;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint qb = group.x * 32;
    const uint base = group.y * 1024 * 64;
    threadgroup half qs[32*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    threadgroup float scores[32*32];
    threadgroup half probs[32*32];
    threadgroup float scales[32];
    threadgroup float sums[32];
    float maxval = -INFINITY;
    float sumval = 0.f;
    simdgroup_float8x8 out[8];
    for (uint d=0; d<8; ++d) out[d] = simdgroup_float8x8(0.f);
    for (uint i=tid; i<32*64; i+=128) qs[i] = half(Q[base+qb*64+i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb=0; kb<=qb; kb+=32) {
        for (uint i=tid; i<32*64; i+=128) {
            ks[i] = half(K[base+kb*64+i]);
            vs[i] = half(V[base+kb*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0; j<4; ++j) {
            simdgroup_float8x8 acc(0.f);
            for (uint d=0; d<8; ++d) {
                simdgroup_half8x8 a,b;
                simdgroup_load(a, qs+sg*8*64+d*8, 64);
                simdgroup_load(b, ks+j*8*64+d*8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(acc,a,b,acc);
            }
            simdgroup_store(acc, scores+sg*8*32+j*8, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<32) {
            float m = -INFINITY;
            for (uint j=0;j<32;++j) {
                float s = kb+j<=qb+tid ? scores[tid*32+j]*0.125f : -INFINITY;
                scores[tid*32+j]=s;
                m = max(m,s);
            }
            float nm = max(m,maxval);
            float scale = exp(maxval-nm);
            float sum = 0.f;
            for (uint j=0;j<32;++j) {
                float p=exp(scores[tid*32+j]-nm);
                probs[tid*32+j]=half(p);
                sum+=p;
            }
            sumval = sumval*scale+sum;
            maxval=nm;
            scales[tid]=scale;
            sums[tid]=sumval;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale=scales[sg*8+(lane%8)/2+(lane/16)*4];
        for (uint d=0;d<8;++d) {
            out[d].thread_elements()[0]*=scale;
            out[d].thread_elements()[1]*=scale;
            for (uint j=0;j<4;++j) {
                simdgroup_half8x8 a,b;
                simdgroup_load(a,probs+sg*8*32+j*8,32);
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(out[d],a,b,out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float den=sums[sg*8+(lane%8)/2+(lane/16)*4];
    for (uint d=0;d<8;++d) {
        out[d].thread_elements()[0]/=den;
        out[d].thread_elements()[1]/=den;
        simdgroup_store(out[d],O+base+(qb+sg*8)*64+d*8,64);
    }
}
'''
_lib = torch.mps.compile_shader(_src)

def kernel(q, k, v):
    if q.shape != (8,8,1024,64):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True)
    out = torch.empty_like(q)
    _lib.attention(q,k,v,out,threads=(32*128,64,1),group_size=(128,1,1))
    return out
