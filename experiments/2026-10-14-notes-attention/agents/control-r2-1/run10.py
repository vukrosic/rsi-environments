import torch

_R, _C = 128, 32

_SOURCE = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint R = TILE_R;
constant uint C = TILE_C;
constant uint NT = TILE_R*4;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qs[R*64];
    threadgroup half ks[64*C];
    threadgroup half vs[C*64];
    threadgroup half ps[R*C];
    uint base = group.y*1024*64;
    uint row0 = group.x*R;
    uint row = sg*8 + (lane/2)%4 + (lane/16)*4;
    uint col = (lane%2)*2 + ((lane/8)%2)*4;
    for (uint i=tid; i<R*64; i+=NT)
        qs[i] = half(Q[base+(row0+i/64)*64+i%64]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qm[8];
    for (uint d=0; d<8; ++d)
        simdgroup_load(qm[d], qs+sg*8*64+d*8, 64);
    simdgroup_float8x8 out[8];
    for (uint d=0; d<8; ++d)
        out[d] = make_filled_simdgroup_matrix<float,8,8>(0.f);
    float m = -INFINITY, l = 0.f;
    for (uint start=0; start<row0+R; start+=C) {
        for (uint i=tid; i<C*64; i+=NT) {
            uint r=i/64, d=i%64;
            ks[d*C+r] = half(K[base+(start+r)*64+d]);
            vs[i] = half(V[base+(start+r)*64+d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[C/8];
        for (uint b=0; b<C/8; ++b) {
            scores[b] = make_filled_simdgroup_matrix<float,8,8>(0.f);
            for (uint d=0; d<8; ++d) {
                simdgroup_half8x8 km;
                simdgroup_load(km, ks+d*8*C+b*8, C);
                simdgroup_multiply_accumulate(scores[b], qm[d], km, scores[b]);
            }
        }
        float mx=-INFINITY;
        for (uint b=0; b<C/8; ++b) {
            for (uint e=0; e<2; ++e) {
                float x = scores[b].thread_elements()[e]*0.125f;
                if (start+b*8+col+e > row0+row) x=-INFINITY;
                scores[b].thread_elements()[e]=x;
                mx=max(mx,x);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        float nm=max(m,mx), alpha=exp(m-nm), sum=0.f;
        for (uint b=0; b<C/8; ++b) {
            simdgroup_half8x8 p;
            for (uint e=0; e<2; ++e) {
                float x=exp(scores[b].thread_elements()[e]-nm);
                sum+=x;
                p.thread_elements()[e]=half(x);
            }
            simdgroup_store(p, ps+sg*8*C+b*8, C);
        }
        sum+=simd_shuffle_xor(sum,1);
        sum+=simd_shuffle_xor(sum,8);
        l=l*alpha+sum;
        m=nm;
        for (uint d=0; d<8; ++d) {
            out[d].thread_elements()[0]*=alpha;
            out[d].thread_elements()[1]*=alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b=0; b<C/8; ++b) {
            simdgroup_half8x8 p;
            simdgroup_load(p, ps+sg*8*C+b*8, C);
            for (uint d=0; d<8; ++d) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vs+b*8*64+d*8, 64);
                simdgroup_multiply_accumulate(out[d], p, vm, out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d=0; d<8; ++d)
        for (uint e=0; e<2; ++e)
            O[base+(row0+row)*64+d*8+col+e]=out[d].thread_elements()[e]/l;
}
'''
_LIB = torch.mps.compile_shader(_SOURCE.replace("TILE_R", str(_R)).replace("TILE_C", str(_C)))


def kernel(q, k, v):
    o = torch.empty_like(q)
    _LIB.attention(q, k, v, o, threads=(4096, 64, 1), group_size=(_R*4, 1, 1))
    return o
