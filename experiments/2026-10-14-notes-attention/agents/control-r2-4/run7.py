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
    const uint qb = group.x * 64;
    const uint base = group.y * 1024 * 64;
    threadgroup half qs[64*64];
    threadgroup half ks[64*64];
    threadgroup half vs[64*64];
    float maxval = -INFINITY;
    float sumval = 0.f;
    simdgroup_float8x8 out[8];
    for (uint d=0; d<8; ++d) out[d] = simdgroup_float8x8(0.f);
    for (uint i=tid; i<64*64; i+=256) qs[i] = half(Q[base+qb*64+i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kb=0; kb<=qb; kb+=64) {
        for (uint i=tid; i<64*64; i+=256) {
            ks[i] = half(K[base+kb*64+i]);
            vs[i] = half(V[base+kb*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc[8];
        for (uint j=0;j<8;++j) acc[j]=simdgroup_float8x8(0.f);
        for (uint d=0;d<8;++d) {
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d*8,64);
            for (uint j=0;j<8;++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
            }
        }
        const uint row=sg*8+(lane%8)/2+(lane/16)*4;
        const uint col=(lane%2)*2+((lane/8)%2)*4;
        float m=-INFINITY;
        for (uint j=0;j<8;++j) {
            for (uint e=0;e<2;++e) {
                float s=kb+j*8+col+e<=qb+row ? acc[j].thread_elements()[e]*0.125f : -INFINITY;
                acc[j].thread_elements()[e]=s;
                m=max(m,s);
            }
        }
        m=max(m,simd_shuffle_xor(m,1));
        m=max(m,simd_shuffle_xor(m,8));
        float nm=max(m,maxval);
        float scale=exp(maxval-nm);
        float sum=0.f;
        simdgroup_half8x8 probs[8];
        for (uint j=0;j<8;++j) {
            for (uint e=0;e<2;++e) {
                float p=exp(acc[j].thread_elements()[e]-nm);
                probs[j].thread_elements()[e]=half(p);
                sum+=p;
            }
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        sumval=sumval*scale+sum;
        maxval=nm;
        for (uint d=0;d<8;++d) {
            out[d].thread_elements()[0]*=scale;
            out[d].thread_elements()[1]*=scale;
            for (uint j=0;j<8;++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(out[d],probs[j],b,out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float den=sumval;
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
    _lib.attention(q,k,v,out,threads=(16*256,64,1),group_size=(256,1,1))
    return out
