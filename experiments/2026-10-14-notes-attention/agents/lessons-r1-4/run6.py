"""Causal attention with half precision matrix products and fused softmax."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define BQ 64
#define BK 64
#define NT (BQ*4)
kernel void attention(device const half* q [[buffer(0)]],
                      device const half* k [[buffer(1)]],
                      device const half* v [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[BQ*64];
    threadgroup half ks[BK*64];
    threadgroup half vs[BK*64];
    threadgroup half ps[BQ*BK];
    uint head = group.y * 1024 * 64;
    uint qr = group.x * BQ;
    uint rr = (lane%8)/2 + (lane/16)*4;
    uint cc = (lane%2)*2 + ((lane/8)%2)*4;
    uint row = qr + sg*8 + rr;
    for(uint i=tid; i<BQ*64; i+=NT) qs[i]=q[head+qr*64+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    for(uint d=0;d<8;++d) simdgroup_load(qm[d],qs+sg*8*64+d*8,64);
    simdgroup_float8x8 acc[8];
    for(uint d=0;d<8;++d) acc[d]=simdgroup_float8x8(0);
    float m=-INFINITY;
    float l=0;
    for(uint kb=0;kb<qr+BQ;kb+=BK) {
        for(uint i=tid;i<BK*64;i+=NT) {
            ks[i]=k[head+kb*64+i];
            vs[i]=v[head+kb*64+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sm[BK/8];
        for(uint j=0;j<BK/8;++j) sm[j]=simdgroup_float8x8(0);
        for(uint d=0;d<8;++d) {
            for(uint j=0;j<BK/8;++j) {
                simdgroup_half8x8 km;
                simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(sm[j],qm[d],km,sm[j]);
            }
        }
        float mx=-INFINITY;
        for(uint j=0;j<BK/8;++j) {
            for(uint e=0;e<2;++e) {
                float x=sm[j].thread_elements()[e]*0.125f;
                if(kb+j*8+cc+e>row) x=-INFINITY;
                sm[j].thread_elements()[e]=x;
                mx=max(mx,x);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        float nm=max(m,mx);
        float alpha=fast::exp(m-nm);
        float sum=0;
        for(uint j=0;j<BK/8;++j) {
            simdgroup_half8x8 pm;
            for(uint e=0;e<2;++e) {
                float p=fast::exp(sm[j].thread_elements()[e]-nm);
                pm.thread_elements()[e]=half(p);
                sum+=p;
            }
            simdgroup_store(pm,ps+sg*8*BK+j*8,BK);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        l=l*alpha+sum;
        m=nm;
        for(uint d=0;d<8;++d) {
            acc[d].thread_elements()[0]*=alpha;
            acc[d].thread_elements()[1]*=alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<BK/8;++j) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm,ps+sg*8*BK+j*8,BK);
            for(uint d=0;d<8;++d) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm,vs+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(acc[d],pm,vm,acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint d=0;d<8;++d) {
        acc[d].thread_elements()[0]/=l;
        acc[d].thread_elements()[1]/=l;
        simdgroup_store(acc[d],out+head+(qr+sg*8)*64+d*8,64);
    }
}
""")

def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.attention(qh, kh, vh, out, threads=(16*256,64), group_size=(256,1))
    return out
