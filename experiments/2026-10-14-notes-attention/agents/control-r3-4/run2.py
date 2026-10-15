import torch

_source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half qs[32*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    threadgroup float scores[32*32];
    threadgroup half probs[32*32];
    threadgroup float outs[32*64];
    uint base = gid.y*1024*64;
    uint qbase = gid.x*32;
    for(uint i=tid;i<32*64;i+=128) qs[i]=half(Q[base+qbase*64+i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[8];
    simdgroup_float8x8 acc[8];
    for(uint d=0;d<8;d++) {
        simdgroup_load(qf[d],qs+sg*8*64+d*8,64);
        acc[d]=make_filled_simdgroup_matrix<float,8>(0.0f);
    }
    float maximum=-INFINITY;
    float denominator=0.0f;
    uint row=sg*8+lane/4;
    for(uint start=0;start<qbase+32;start+=32) {
        for(uint i=tid;i<32*64;i+=128) {
            ks[i]=half(K[base+start*64+i]);
            vs[i]=half(V[base+start*64+i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sf[4];
        for(uint c=0;c<4;c++) {
            sf[c]=make_filled_simdgroup_matrix<float,8>(0.0f);
            for(uint d=0;d<8;d++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf,ks+c*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(sf[c],qf[d],kf,sf[c]);
            }
            simdgroup_store(sf[c],scores+sg*8*32+c*8,32);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float vals[8];
        float mx=-INFINITY;
        for(uint c=0;c<8;c++) {
            uint col=lane%4+c*4;
            float s=start+col<=qbase+row ? scores[row*32+col]*0.125f : -INFINITY;
            vals[c]=s;
            mx=max(mx,s);
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,2));
        float nm=max(maximum,mx);
        float alpha=fast::exp(maximum-nm);
        float sm=0.0f;
        for(uint c=0;c<8;c++) {
            float p=fast::exp(vals[c]-nm);
            probs[row*32+lane%4+c*4]=half(p);
            sm+=p;
        }
        sm+=simd_shuffle_xor(sm,1);
        sm+=simd_shuffle_xor(sm,2);
        denominator=denominator*alpha+sm;
        maximum=nm;
        for(uint d=0;d<8;d++) {
            simdgroup_store(acc[d],outs+sg*8*64+d*8,64);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=lane%4;d<64;d+=4) outs[row*64+d]*=alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=0;d<8;d++) {
            simdgroup_load(acc[d],outs+sg*8*64+d*8,64);
            for(uint c=0;c<4;c++) {
                simdgroup_half8x8 pf,vf;
                simdgroup_load(pf,probs+sg*8*32+c*8,32);
                simdgroup_load(vf,vs+c*8*64+d*8,64);
                simdgroup_multiply_accumulate(acc[d],pf,vf,acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint d=0;d<8;d++) simdgroup_store(acc[d],outs+sg*8*64+d*8,64);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane%4;d<64;d+=4) O[base+(qbase+row)*64+d]=outs[row*64+d]/denominator;
}
"""
_lib = torch.mps.compile_shader(_source)


def kernel(q, k, v):
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out, threads=(32*128,64,1), group_size=(128,1,1))
    return out
