"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch

_source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float* Q, device const float* K,
                      device const float* V, device float* O,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[32*64], ks[32*64], vs[32*64], ps[32*32];
    threadgroup float scores[32*32], mx[32], denom[32], alpha[32];
    uint base=group.y*1024*64, start=group.x*32;
    for(uint i=tid;i<32*64;i+=128) qs[i]=half(Q[base+start*64+i]);
    if(tid<32){mx[tid]=-INFINITY;denom[tid]=0;}
    simdgroup_float8x8 out[8];
    for(uint d=0;d<8;d++) out[d]=simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<=start;block+=32){
        for(uint i=tid;i<32*64;i+=128){ks[i]=half(K[base+block*64+i]);vs[i]=half(V[base+block*64+i]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
        for(uint j=0;j<4;j++) s[j]=simdgroup_float8x8(0);
        for(uint d=0;d<8;d++){
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d*8,64);
            for(uint j=0;j<4;j++){
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(s[j],a,b,s[j]);
            }
        }
        for(uint j=0;j<4;j++) simdgroup_store(s[j],scores+sg*8*32+j*8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32){
            float m=mx[tid];
            for(uint j=0;j<32;j++) if(block+j<=start+tid) m=max(m,scores[tid*32+j]*0.125f);
            float a=exp(mx[tid]-m), sum=0;
            for(uint j=0;j<32;j++){
                float p=block+j<=start+tid?exp(scores[tid*32+j]*0.125f-m):0;
                ps[tid*32+j]=half(p);sum+=p;
            }
            alpha[tid]=a;denom[tid]=denom[tid]*a+sum;mx[tid]=m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale=alpha[sg*8+((lane&6)/2+(lane&16)/4)];
        for(uint d=0;d<8;d++){
            out[d].thread_elements()[0]*=scale;
            out[d].thread_elements()[1]*=scale;
        }
        for(uint j=0;j<4;j++){
            simdgroup_half8x8 p;
            simdgroup_load(p,ps+sg*8*32+j*8,32);
            for(uint d=0;d<8;d++){
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(out[d],p,b,out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1.0f/denom[sg*8+((lane&6)/2+(lane&16)/4)];
    for(uint d=0;d<8;d++){
        out[d].thread_elements()[0]*=inv;out[d].thread_elements()[1]*=inv;
        simdgroup_store(out[d],O+base+(start+sg*8)*64+d*8,64);
    }
}
"""
_lib = torch.mps.compile_shader(_source)

def kernel(q, k, v):
    o = torch.empty_like(q)
    _lib.attention(q, k, v, o, threads=(32*128,64,1), group_size=(128,1,1))
    return o
