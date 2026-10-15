"""Fused tiled causal attention for MPS."""
import torch

_BQ = 128
_BK = 64
_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint BQ = 128;
constant uint BK = 64;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half ks[BK*64];
    threadgroup half vs[BK*64];
    threadgroup half scores[BQ*BK];
    uint tid=sg*32+lane;
    uint row0 = group.x * BQ + sg * 8;
    uint base = group.y * 1024 * 64;
    uint r = (lane % 8) / 2 + (lane / 16) * 4;
    uint c = lane % 2 + ((lane / 8) % 2) * 2;
    simdgroup_half8x8 query[8];
    #pragma unroll
    for(uint d=0;d<8;d++) {
        half2 h=half2(*reinterpret_cast<device const float2*>(Q+base+(row0+r)*64+d*8+c*2));
        query[d].thread_elements()[0]=h.x;
        query[d].thread_elements()[1]=h.y;
    }
    float m = -INFINITY, l = 0.0f;
    simdgroup_float8x8 out[8];
    #pragma unroll
    for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
    for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
        for(uint i=tid;i<BK*64;i+=512) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<BK/8;j++) {
            simdgroup_float8x8 acc(0.0f);
            #pragma unroll
            for(uint d=0;d<64;d+=8) {
                simdgroup_half8x8 a=query[d/8],b;
                simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                simdgroup_multiply_accumulate(acc,a,b,acc);
            }
            simdgroup_half8x8 h;
            h.thread_elements()[0]=half(acc.thread_elements()[0]*0.125f);
            h.thread_elements()[1]=half(acc.thread_elements()[1]*0.125f);
            simdgroup_store(h,scores+sg*8*BK+j*8,BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float mx = -INFINITY;
        for(uint j=c;j<BK;j+=4) {
            float s=kb+j<=row0+r ? float(scores[(sg*8+r)*BK+j]) : -INFINITY;
            scores[(sg*8+r)*BK+j]=half(s);
            mx=max(mx,s);
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        float nm=max(m,mx), alpha=exp(m-nm), sum=0;
        for(uint j=c;j<BK;j+=4) {
            float p=exp(float(scores[(sg*8+r)*BK+j])-nm);
            sum+=p;
            scores[(sg*8+r)*BK+j]=half(p);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        l=l*alpha+sum; m=nm;
        #pragma unroll
        for(uint j=0;j<8;j++) {
            out[j].thread_elements()[0]*=alpha;
            out[j].thread_elements()[1]*=alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<8;j++) {
            for(uint d=0;d<BK;d+=8) {
                simdgroup_half8x8 a;
                simdgroup_load(a,scores+sg*8*BK+d,BK);
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+d*64+j*8,64);
                simdgroup_multiply_accumulate(out[j],a,b,out[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint j=0;j<8;j++) {
        out[j].thread_elements()[0]/=l;
        out[j].thread_elements()[1]/=l;
        simdgroup_store(out[j],O+base+row0*64+j*8,64);
    }
}
''')

def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out,
                   threads=(8*512,64,1), group_size=(512,1,1))
    return out
