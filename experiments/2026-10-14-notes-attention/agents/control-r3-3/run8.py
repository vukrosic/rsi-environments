"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch

_source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define BQ 32
#define BK 32
#define NT (BQ*4)
#define NK (BK/8)
kernel void attention(device const float* Q, device const float* K,
                      device const float* V, device float* O,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[BQ*64], ks[BK*64], vs[BK*64];
    uint base=group.y*1024*64, start=group.x*BQ;
    for(uint i=tid;i<BQ*64;i+=NT) qs[i]=half(Q[base+start*64+i]);
    uint row=sg*8+((lane&6)/2+(lane&16)/4);
    uint col=(lane&1)*2+(lane&8)/2;
    float mx=-INFINITY, denom=0;
    simdgroup_half8x8 out[8];
    #pragma unroll
    for(uint d=0;d<8;d++) out[d]=simdgroup_half8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<start+BQ;block+=BK){
        for(uint i=tid;i<BK*64;i+=NT){ks[i]=half(K[base+block*64+i]);vs[i]=half(V[base+block*64+i]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[NK];
        #pragma unroll
        for(uint j=0;j<NK;j++) s[j]=simdgroup_float8x8(0);
        #pragma unroll
    for(uint d=0;d<8;d++){
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d*8,64);
            #pragma unroll
        for(uint j=0;j<NK;j++){
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(s[j],a,b,s[j]);
            }
        }
        float m=mx;
        #pragma unroll
        for(uint j=0;j<NK;j++){
            s[j].thread_elements()[0]=block+j*8+col<=start+row?s[j].thread_elements()[0]*0.125f:-INFINITY;
            s[j].thread_elements()[1]=block+j*8+col+1<=start+row?s[j].thread_elements()[1]*0.125f:-INFINITY;
            m=max(m,float(max(s[j].thread_elements()[0],s[j].thread_elements()[1])));
        }
        m=max(m,simd_shuffle_xor(m,1));m=max(m,simd_shuffle_xor(m,8));
        float scale=exp(mx-m), sum=0;
        simdgroup_half8x8 p[NK];
        #pragma unroll
        for(uint j=0;j<NK;j++){
            float a=exp(s[j].thread_elements()[0]-m), b=exp(s[j].thread_elements()[1]-m);
            p[j].thread_elements()[0]=half(a);p[j].thread_elements()[1]=half(b);sum+=a+b;
        }
        sum+=simd_shuffle_xor(sum,1);sum+=simd_shuffle_xor(sum,8);
        denom=denom*scale+sum;mx=m;
        #pragma unroll
    for(uint d=0;d<8;d++){
            out[d].thread_elements()[0]*=scale;
            out[d].thread_elements()[1]*=scale;
        }
        #pragma unroll
        for(uint j=0;j<NK;j++){
            #pragma unroll
    for(uint d=0;d<8;d++){
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(out[d],p[j],b,out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv=1.0f/denom;
    #pragma unroll
    for(uint d=0;d<8;d++){
        simdgroup_float8x8 result;
        result.thread_elements()[0]=float(out[d].thread_elements()[0])*inv;result.thread_elements()[1]=float(out[d].thread_elements()[1])*inv;
        simdgroup_store(result,O+base+(start+sg*8)*64+d*8,64);
    }
}
"""
_lib = torch.mps.compile_shader(_source)

def kernel(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
