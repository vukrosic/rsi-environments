import torch
import torch.nn.functional as F

_source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const half* q [[buffer(0)]],
                      device const half* k [[buffer(1)]],
                      device const half* v [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kv[32*72];
    threadgroup half vt[32*72];
    threadgroup half weight[64*40];
    uint base = gid.y*1024*64;
    uint row = gid.x*64;
    uint myrow = sg*8+(lane&6)/2+(lane&16)/4;
    uint mycol = (lane&1)*2+(lane&8)/2;
    simdgroup_half8x8 queries[8];
    #pragma unroll
    for(uint d=0;d<8;d++) simdgroup_load(queries[d],q+base+(row+sg*8)*64+d*8,64);
    float denom=0.0f;
    float maximum=-INFINITY;
    simdgroup_float8x8 output[8];
    #pragma unroll
    for(uint j=0;j<8;j++) output[j]=simdgroup_float8x8(0.0f);
    #pragma unroll
    for(uint start=0;start<row+64;start+=32) {
        #pragma unroll
    for(uint i=tid;i<32*64;i+=256) {
            kv[(i/64)*72+i%64]=k[base+start*64+i];
            vt[(i/64)*72+i%64]=v[base+start*64+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[4];
        #pragma unroll
    for(uint j=0;j<4;j++) scores[j]=simdgroup_float8x8(0.0f);
        #pragma unroll
    for(uint d=0;d<8;d++) {
            #pragma unroll
    for(uint j=0;j<4;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,kv+j*8*72+d*8,72,ulong2(0),true);
                simdgroup_multiply_accumulate(scores[j],queries[d],b,scores[j]);
            }
        }
        float localmax=-INFINITY;
        #pragma unroll
    for(uint j=0;j<4;j++) {
            #pragma unroll
    for(uint e=0;e<2;e++) {
                float s=start+j*8+mycol+e<=row+myrow ? scores[j].thread_elements()[e]*0.125f : -INFINITY;
                scores[j].thread_elements()[e]=s;
                localmax=max(localmax,s);
            }
        }
        localmax=max(localmax,simd_shuffle_xor(localmax,1));
        localmax=max(localmax,simd_shuffle_xor(localmax,8));
        float newmax=max(maximum,localmax);
        float alpha=fast::exp(maximum-newmax);
        float sum=0.0f;
        #pragma unroll
    for(uint j=0;j<4;j++) {
            simdgroup_half8x8 w;
            #pragma unroll
    for(uint e=0;e<2;e++) {
                half value=half(fast::exp(scores[j].thread_elements()[e]-newmax));
                w.thread_elements()[e]=value;
                sum+=float(value);
                output[j].thread_elements()[e]*=alpha;
            }
            simdgroup_store(w,weight+sg*8*40+j*8,40);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        denom=denom*alpha+sum;
        maximum=newmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
    for(uint d=0;d<4;d++) {
            simdgroup_half8x8 a;
            simdgroup_load(a,weight+sg*8*40+d*8,40);
            #pragma unroll
    for(uint j=0;j<8;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,vt+d*8*72+j*8,72);
                simdgroup_multiply_accumulate(output[j],a,b,output[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint j=0;j<8;j++) {
        output[j].thread_elements()[0]/=denom;
        output[j].thread_elements()[1]/=denom;
        simdgroup_store(output[j],out+base+(row+sg*8)*64+j*8,64);
    }
}
'''
_lib = torch.mps.compile_shader(_source)

def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = torch.empty_like(q)
    _lib.attention(q.half(), k.half(), v.half(), out,
                   threads=(16*256,64,1), group_size=(256,1,1))
    return out
