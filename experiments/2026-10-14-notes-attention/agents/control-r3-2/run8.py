import torch
import torch.nn.functional as F

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
#define BQ 32
#define BK 32
kernel void flash(device const half* q [[buffer(0)]],
                  device const half* k [[buffer(1)]],
                  device const half* v [[buffer(2)]],
                  device float* out [[buffer(3)]],
                  uint2 gid [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    uint qb=gid.x*BQ, bh=gid.y;
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
    float oldm=-INFINITY, denom=0.0f;
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
        float m=-INFINITY;
        #pragma unroll
        for(uint j=0;j<BK/4;j++) {
            uint c=col+(j/2)*8+j%2;
            x[j]=kb+c<=qb+row ? sm[j/2].thread_elements()[j%2]*0.125f : -INFINITY;
            m=max(m,x[j]);
        }
        m=max(m,simd_shuffle_xor(m,1));
        m=max(m,simd_shuffle_xor(m,8));
        m=max(m,oldm);
        float alpha=fast::exp(oldm-m);
        float z=0;
        #pragma unroll
        for(uint j=0;j<BK/4;j++) {
            float p=fast::exp(x[j]-m);
            x[j]=p;
            z+=p;
        }
        z+=simd_shuffle_xor(z,1);
        z+=simd_shuffle_xor(z,8);
        denom=denom*alpha+z;
        oldm=m;
        #pragma unroll
        for(uint d=0;d<8;d++) {
            om[d].thread_elements()[0]*=alpha;
            om[d].thread_elements()[1]*=alpha;
        }
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
kernel void softmax_causal(device half* s [[buffer(0)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint pos = row % 1024;
    float4 x;
    for (uint j=0;j<4;j++) {
        uint c = tid+j*256;
        x[j] = c<=pos ? float(s[row*1024+c])*0.125f : -INFINITY;
    }
    float m = simd_max(max(max(x.x,x.y),max(x.z,x.w)));
    if(lane==0) scratch[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = simd_max(lane<8?scratch[lane]:-INFINITY);
    float4 e = fast::exp(x-m);
    float z = simd_sum(e.x+e.y+e.z+e.w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) scratch[sg]=z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    z = simd_sum(lane<8?scratch[lane]:0.0f);
        #pragma unroll
    for(uint j=0;j<4;j++) s[row*1024+tid+j*256]=half(e[j]/z);
}
""")


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    out = torch.empty_like(q)
    _lib.flash(qh, kh, vh, out, threads=(32*128, 64), group_size=(128, 1))
    return out
