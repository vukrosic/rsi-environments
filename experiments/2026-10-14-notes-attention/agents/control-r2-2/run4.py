import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint2 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[32*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    threadgroup float scores[32*32];
    threadgroup half ps[32*32];
    threadgroup float sums[32];
    threadgroup float outs[32*64];
    uint query0 = gid.x*32;
    uint base = gid.y*1024*64;
    for(uint i=tid; i<32*64; i+=128) qs[i]=half(Q[base+query0*64+i]);
    if(tid<32) sums[tid]=0;
    simdgroup_float8x8 acc[8];
    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kb=0; kb<=query0; kb+=32) {
        for(uint i=tid; i<32*64; i+=128) {
            ks[i]=half(K[base+kb*64+i]);
            vs[i]=half(V[base+kb*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
        for(uint j=0;j<4;++j) s[j]=simdgroup_float8x8(0);
        for(uint d=0;d<8;++d) {
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d*8,64);
            for(uint j=0;j<4;++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(s[j],a,b,s[j]);
            }
        }
        for(uint j=0;j<4;++j) simdgroup_store(s[j],scores+sg*8*32+j*8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32) {
            float sum=0;
            for(uint j=0;j<32;++j) {
                float p = kb+j<=query0+tid ? exp(scores[tid*32+j]*0.125f-2.0f) : 0.0f;
                ps[tid*32+j]=half(p);
                sum+=p;
            }
            sums[tid]+=sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<4;++j) {
            simdgroup_half8x8 a;
            simdgroup_load(a,ps+sg*8*32+j*8,32);
            for(uint d=0;d<8;++d) {
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(acc[d],a,b,acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint d=0;d<8;++d) simdgroup_store(acc[d],outs+sg*8*64+d*8,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=128) O[base+query0*64+i]=outs[i]/sums[i/64];
}
""")

def kernel(q,k,v):
    o=torch.empty_like(q)
    _lib.attention(q,k,v,o,threads=(32*128,64,1),group_size=(128,1,1))
    return o
