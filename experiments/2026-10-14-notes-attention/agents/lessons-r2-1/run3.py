"""Fused tiled causal attention for MPS."""
import torch

_BQ = 64
_BK = 64
_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint BQ = 64;
constant uint BK = 64;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[BQ*64];
    threadgroup half ks[BK*64];
    threadgroup half vs[BK*64];
    uint tid=sg*32+lane;
    uint row0 = group.x * BQ + sg * 8;
    uint base = group.y * 1024 * 64;
    uint r = (lane % 8) / 2 + (lane / 16) * 4;
    uint c = lane % 2 + ((lane / 8) % 2) * 2;
    for(uint i=tid;i<BQ*64;i+=256) qs[i]=half(Q[base+group.x*BQ*64+i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY, l = 0.0f;
    simdgroup_float8x8 out[8];
    for (uint j=0;j<8;j++) out[j] = simdgroup_float8x8(0.0f);
    for (uint kb=0;kb<group.x*BQ+BQ;kb+=BK) {
        for(uint i=tid;i<BK*64;i+=256) {ks[i]=half(K[base+kb*64+i]);vs[i]=half(V[base+kb*64+i]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc[8];
        for(uint j=0;j<8;j++) acc[j] = simdgroup_float8x8(0.0f);
        for(uint d=0;d<64;d+=8) {
            simdgroup_half8x8 a;
            simdgroup_load(a,qs+sg*8*64+d,64);
            for(uint j=0;j<BK/8;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,ks+j*8*64+d,64,ulong2(0),true);
                simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
            }
        }
        float mx = -INFINITY;
        for(uint j=0;j<BK/8;j++) {
            for(uint e=0;e<2;e++) {
                uint col=j*8+c*2+e;
                float s=kb+col<=row0+r ? acc[j].thread_elements()[e]*0.125f : -INFINITY;
                acc[j].thread_elements()[e]=s;
                mx=max(mx,s);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        float nm=max(m,mx), alpha=exp(m-nm), sum=0;
        simdgroup_half8x8 probs[BK/8];
        for(uint j=0;j<BK/8;j++) {
            for(uint e=0;e<2;e++) {
                float p=exp(acc[j].thread_elements()[e]-nm);
                sum+=p;
                probs[j].thread_elements()[e]=half(p);
            }
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        l=l*alpha+sum; m=nm;
        for(uint j=0;j<8;j++) {
            out[j].thread_elements()[0]*=alpha;
            out[j].thread_elements()[1]*=alpha;
        }
        for(uint d=0;d<BK;d+=8) {
            simdgroup_half8x8 a=probs[d/8];
            for(uint j=0;j<8;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,vs+d*64+j*8,64);
                simdgroup_multiply_accumulate(out[j],a,b,out[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
                   threads=(16*256,64,1), group_size=(256,1,1))
    return out
