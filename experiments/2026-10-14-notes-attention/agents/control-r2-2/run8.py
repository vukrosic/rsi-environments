import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const half* Q [[buffer(0)]],
                      device const half* K [[buffer(1)]],
                      device const half* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint2 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half qs[32*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    uint query0 = gid.x*32;
    uint base = gid.y*1024*64;
    for(uint i=tid; i<32*64; i+=128) qs[i]=half(Q[base+query0*64+i]);
    float norm=0;
    uint row=(lane%8)/2+4*(lane/16);
    uint col=(lane%2)*2+4*((lane/8)%2);
    simdgroup_float8x8 acc[8];
    #pragma unroll
    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kb=0; kb<query0+32; kb+=32) {
        for(uint i=tid; i<32*64; i+=128) {
            ks[i]=half(K[base+kb*64+i]);
            vs[i]=half(V[base+kb*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
        #pragma unroll
        for(uint j=0;j<4;++j) s[j]=simdgroup_float8x8(0);
        #pragma unroll
        for(uint d=0;d<8;++d) {
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d*8,64);
            #pragma unroll
            for(uint j=0;j<4;++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(s[j],a,b,s[j]);
            }
        }
        simdgroup_half8x8 probs[4];
        float sum=0;
        #pragma unroll
        for(uint j=0;j<4;++j) {
            float p0=kb+j*8+col<=query0+sg*8+row ? exp(s[j].thread_elements()[0]*0.125f-2.0f) : 0.0f;
            float p1=kb+j*8+col+1<=query0+sg*8+row ? exp(s[j].thread_elements()[1]*0.125f-2.0f) : 0.0f;
            probs[j].thread_elements()[0]=half(p0);
            probs[j].thread_elements()[1]=half(p1);
            sum+=p0+p1;
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        norm+=sum;
        #pragma unroll
        for(uint j=0;j<4;++j) {
            #pragma unroll
            for(uint d=0;d<8;++d) {
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(acc[d],probs[j],b,acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1.0f/norm;
    #pragma unroll
    for(uint d=0;d<8;++d) {
        acc[d].thread_elements()[0]*=inv;
        acc[d].thread_elements()[1]*=inv;
        simdgroup_store(acc[d],O+base+(query0+sg*8)*64+d*8,64);
    }

}
""")

def kernel(q,k,v):
    o=torch.empty_like(q)
    _lib.attention(q.half(),k.half(),v.half(),o,threads=(32*128,64,1),group_size=(128,1,1))
    return o
