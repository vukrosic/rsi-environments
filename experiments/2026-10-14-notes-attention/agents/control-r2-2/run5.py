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
    threadgroup half qs[64*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    threadgroup float scores[64*32];
    threadgroup half ps[64*32];
    threadgroup float sums[64];
    uint query0 = gid.x*64;
    uint base = gid.y*1024*64;
    for(uint i=tid; i<64*64; i+=256) qs[i]=half(Q[base+query0*64+i]);
    if(tid<64) sums[tid]=0;
    simdgroup_float8x8 acc[8];
    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kb=0; kb<query0+64; kb+=32) {
        for(uint i=tid; i<32*64; i+=256) {
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
        uint row=tid/4;
        uint col=(tid%4)*8;
        float sum=0;
        for(uint j=0;j<8;++j) {
            float p=kb+col+j<=query0+row ? exp(scores[row*32+col+j]*0.125f-2.0f) : 0.0f;
            ps[row*32+col+j]=half(p);
            sum+=p;
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,2);
        if(tid%4==0) sums[row]+=sum;
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
    uint row=sg*8+(lane%8)/2+4*(lane/16);
    float inv=1.0f/sums[row];
    for(uint d=0;d<8;++d) {
        acc[d].thread_elements()[0]*=inv;
        acc[d].thread_elements()[1]*=inv;
        simdgroup_store(acc[d],O+base+(query0+sg*8)*64+d*8,64);
    }

}
""")

def kernel(q,k,v):
    o=torch.empty_like(q)
    _lib.attention(q.half(),k.half(),v.half(),o,threads=(16*256,64,1),group_size=(256,1,1))
    return o
