"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch

_source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const half* Q, device const half* K,
                      device const half* V, device float* O,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[64*64], ks[32*64], vs[32*64], ps[64*32];
    threadgroup float scores[64*32], mx[64], denom[64], alpha[64];
    uint base=group.y*1024*64, start=group.x*64;
    for(uint i=tid;i<64*64;i+=256) qs[i]=half(Q[base+start*64+i]);
    if(tid<64){mx[tid]=-INFINITY;denom[tid]=0;}
    simdgroup_float8x8 out[8];
    for(uint d=0;d<8;d++) out[d]=simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<start+64;block+=32){
        for(uint i=tid;i<32*64;i+=256){ks[i]=half(K[base+block*64+i]);vs[i]=half(V[base+block*64+i]);}
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
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint row=sg*8+((lane&6)/2+(lane&16)/4);
        uint col=(lane&1)*2+(lane&8)/2;
        float m=mx[row];
        for(uint j=col;j<32;j+=8){
            if(block+j<=start+row) m=max(m,scores[row*32+j]*0.125f);
            if(block+j+1<=start+row) m=max(m,scores[row*32+j+1]*0.125f);
        }
        m=max(m,simd_shuffle_xor(m,1));m=max(m,simd_shuffle_xor(m,8));
        float scale=exp(mx[row]-m), sum=0;
        for(uint j=col;j<32;j+=8){
            float p=block+j<=start+row?exp(scores[row*32+j]*0.125f-m):0;
            float p1=block+j+1<=start+row?exp(scores[row*32+j+1]*0.125f-m):0;
            ps[row*32+j]=half(p);ps[row*32+j+1]=half(p1);sum+=p+p1;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,8);
        float total=denom[row]*scale+sum;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if(col==0){denom[row]=total;mx[row]=m;}
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
    _lib.attention(q.half(), k.half(), v.half(), o, threads=(16*256,64,1), group_size=(256,1,1))
    return o
