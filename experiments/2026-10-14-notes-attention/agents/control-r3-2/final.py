"""Tiled Metal attention for the fixed 8 x 8 x 1024 x 64 workload."""

import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
#define BQ 32
#define BK 64
kernel void flash(device const half* q [[buffer(0)]],
                  device const half* k [[buffer(1)]],
                  device const half* v [[buffer(2)]],
                  device float* out [[buffer(3)]],
                  uint2 gid [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    uint qb=gid.x*BQ, bh=gid.y;
    // The two matrix fragments in each lane share a query row.
    uint row=sg*8+(lane/2)%4+(lane/16)*4;
    uint col=(lane%2)*2+((lane/8)%2)*4;
    q += bh*1024*64;
    k += bh*1024*64;
    v += bh*1024*64;
    out += bh*1024*64;
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 om[8];
    #pragma unroll
    for(uint d=0;d<8;d++) {
        simdgroup_load(qm[d],q+(qb+sg*8)*64+d*8,64);
        om[d]=simdgroup_float8x8(0.0f);
    }
    // Accumulate the numerator and denominator without a full score tensor.
    float denom=0.0f;
    for(uint kb=0;kb<qb+BQ;kb+=BK) {
        simdgroup_float8x8 sm[BK/8];
        #pragma unroll
        for(uint j=0;j<BK/8;j++) sm[j]=simdgroup_float8x8(0.0f);
        #pragma unroll
        for(uint d=0;d<8;d++) {
            #pragma unroll
            for(uint j=0;j<BK/8;j++) {
                simdgroup_half8x8 km;
                simdgroup_load(km,k+(kb+j*8)*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(sm[j],qm[d],km,sm[j]);
            }
        }
        float x[BK/4];
        #pragma unroll
        for(uint j=0;j<BK/4;j++) {
            uint c=col+(j/2)*8+j%2;
            x[j]=kb+c<=qb+row ? sm[j/2].thread_elements()[j%2]*0.125f : -INFINITY;
        }
        float z=0;
        #pragma unroll
        for(uint j=0;j<BK/4;j++) {
            float p=fast::exp(x[j]);
            x[j]=p;
            z+=p;
        }
        z+=simd_shuffle_xor(z,1);
        z+=simd_shuffle_xor(z,8);
        denom+=z;
        #pragma unroll
        for(uint j=0;j<BK/8;j++) {
            simdgroup_half8x8 pm;
            pm.thread_elements()[0]=half(x[j*2]);
            pm.thread_elements()[1]=half(x[j*2+1]);
            #pragma unroll
            for(uint d=0;d<8;d++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm,v+(kb+j*8)*64+d*8,64);
                simdgroup_multiply_accumulate(om[d],pm,vm,om[d]);
            }
        }
    }
    #pragma unroll
    for(uint d=0;d<8;d++) {
        om[d].thread_elements()[0]/=denom;
        om[d].thread_elements()[1]/=denom;
        simdgroup_store(om[d],out+(qb+sg*8)*64+d*8,64);
    }
}
""")


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=(32*128, 64), group_size=(128, 1))
    return out
