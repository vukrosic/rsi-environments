import torch

_source = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void softmax_causal(device half* S [[buffer(0)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    float a[4];
    float mx=-INFINITY;
    #pragma unroll
    for(uint j=0;j<4;j++) {
        uint c=tid+j*256;
        a[j]=c<=row%1024 ? float(S[row*1024+c])*0.125f : -INFINITY;
        mx=max(mx,a[j]);
    }
    mx=simd_max(mx);
    if(lane==0) partial[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=partial[0];
    for(uint i=1;i<8;i++) mx=max(mx,partial[i]);
    float sm=0.0f;
    #pragma unroll
    for(uint j=0;j<4;j++) { a[j]=fast::exp(a[j]-mx); sm+=a[j]; }
    sm=simd_sum(sm);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) partial[sg]=sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sm=0.0f;
    for(uint i=0;i<8;i++) sm+=partial[i];
    #pragma unroll
    for(uint j=0;j<4;j++) S[row*1024+tid+j*256]=half(a[j]/sm);
}

kernel void pack(device const float4* Q [[buffer(0)]],
                 device const float4* K [[buffer(1)]],
                 device const float4* V [[buffer(2)]],
                 device half4* X [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    X[i]=half4(Q[i]);
    X[1048576+i]=half4(K[i]);
    X[2097152+i]=half4(V[i]);
}

kernel void attention(device const half* X [[buffer(0)]],
                      device float* O [[buffer(1)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    device const half* Q=X;
    device const half* K=X+4194304;
    device const half* V=X+8388608;
    threadgroup half qs[64*64];
    threadgroup half ks[32*64];
    threadgroup half vs[32*64];
    uint base = gid.y*1024*64;
    uint qbase = gid.x*64;
    for(uint i=tid;i<64*64;i+=256) qs[i]=Q[base+qbase*64+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[8];
    simdgroup_float8x8 acc[8];
    #pragma unroll
    for(uint d=0;d<8;d++) {
        simdgroup_load(qf[d],qs+sg*8*64+d*8,64);
        acc[d]=make_filled_simdgroup_matrix<float,8>(0.0f);
    }
    float maximum=-INFINITY;
    float denominator=0.0f;
    uint row=sg*8+((lane>>1)&3)+(lane>>4)*4;
    uint col0=(lane&1)*2+((lane>>3)&1)*4;
    for(uint start=0;start<qbase+64;start+=32) {
        for(uint i=tid;i<32*64;i+=256) {
            ks[i]=K[base+start*64+i];
            vs[i]=V[base+start*64+i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sf[4];
        #pragma unroll
        for(uint c=0;c<4;c++) {
            sf[c]=make_filled_simdgroup_matrix<float,8>(0.0f);
            #pragma unroll
            for(uint d=0;d<8;d++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf,ks+c*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(sf[c],qf[d],kf,sf[c]);
            }
        }
        float vals[8];
        float mx=-INFINITY;
        #pragma unroll
        for(uint c=0;c<4;c++) {
            #pragma unroll
            for(uint j=0;j<2;j++) {
                uint col=c*8+col0+j;
                float s=start+col<=qbase+row ? sf[c].thread_elements()[j]*0.125f : -INFINITY;
                vals[c*2+j]=s;
                mx=max(mx,s);
            }
        }
        mx=max(mx,simd_shuffle_xor(mx,1));
        mx=max(mx,simd_shuffle_xor(mx,8));
        float nm=max(maximum,mx);
        float alpha=fast::exp(maximum-nm);
        float sm=0.0f;
        simdgroup_half8x8 pf[4];
        #pragma unroll
        for(uint c=0;c<4;c++) {
            #pragma unroll
            for(uint j=0;j<2;j++) {
                float p=fast::exp(vals[c*2+j]-nm);
                pf[c].thread_elements()[j]=half(p);
                sm+=p;
            }
        }
        sm+=simd_shuffle_xor(sm,1);
        sm+=simd_shuffle_xor(sm,8);
        denominator=denominator*alpha+sm;
        maximum=nm;
        #pragma unroll
        for(uint d=0;d<8;d++) {
            acc[d].thread_elements()[0]*=alpha;
            acc[d].thread_elements()[1]*=alpha;
            #pragma unroll
            for(uint c=0;c<4;c++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf,vs+c*8*64+d*8,64);
                simdgroup_multiply_accumulate(acc[d],pf[c],vf,acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint d=0;d<8;d++) {
        acc[d].thread_elements()[0]/=denominator;
        acc[d].thread_elements()[1]/=denominator;
        simdgroup_store(acc[d],O+base+(qbase+sg*8)*64+d*8,64);
    }
}
"""
_lib = torch.mps.compile_shader(_source)


def kernel(q, k, v):
    packed = torch.empty((3, 8, 8, 1024, 64), device=q.device, dtype=torch.float16)
    _lib.pack(q, k, v, packed, threads=1048576, group_size=256)
    s = packed[0] @ packed[1].transpose(-1, -2)
    _lib.softmax_causal(s, threads=65536*256, group_size=256)
    return (s @ packed[2]).float()
