"""Causal attention with half precision matrix products and fused softmax."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void causal_scores(device const half* q [[buffer(0)]],
                          device const half* k [[buffer(1)]],
                          device half* scores [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    if(group.y > group.x) return;
    threadgroup half qs[64*64];
    threadgroup half ks[64*64];
    uint head=group.z*1024*64;
    uint qr=group.x*64;
    uint kr=group.y*64;
    for(uint i=tid;i<64*64;i+=256) {
        qs[i]=q[head+qr*64+i];
        ks[i]=k[head+kr*64+i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc[8];
    for(uint j=0;j<8;++j) acc[j]=simdgroup_float8x8(0);
    for(uint d=0;d<8;++d) {
        simdgroup_half8x8 qm;
        simdgroup_load(qm,qs+sg*8*64+d*8,64);
        for(uint j=0;j<8;++j) {
            simdgroup_half8x8 km;
            simdgroup_load(km,ks+j*8*64+d*8,64,ulong2(0),true);
            simdgroup_multiply_accumulate(acc[j],qm,km,acc[j]);
        }
    }
    uint base=group.z*1024*1024+(qr+sg*8)*1024+kr;
    for(uint j=0;j<8;++j) {
        simdgroup_half8x8 result;
        result.thread_elements()[0]=half(acc[j].thread_elements()[0]);
        result.thread_elements()[1]=half(acc[j].thread_elements()[1]);
        simdgroup_store(result,scores+base+j*8,1024);
    }
}
kernel void causal_softmax(device const half* scores [[buffer(0)]],
                           device half* probs [[buffer(1)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint pos = row % 1024;
    uint base = row * 1024;
    float x[4];
    float mx = -INFINITY;
    for (uint i=0; i<4; ++i) {
        uint col = tid + i*256;
        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
        mx = max(mx,x[i]);
    }
    mx = simd_max(mx);
    if (lane==0) scratch[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = tid<8 ? scratch[tid] : -INFINITY;
    mx = simd_max(mx);
    if (tid==0) scratch[0]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=scratch[0];
    float sum=0;
    for (uint i=0; i<4; ++i) { x[i]=fast::exp(x[i]-mx); sum+=x[i]; }
    sum=simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) scratch[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=tid<8?scratch[tid]:0;
    sum=simd_sum(sum);
    if(tid==0) scratch[0]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.0f/scratch[0];
    for(uint i=0;i<4;++i) probs[base+tid+i*256]=half(x[i]*inv);
}
""")

def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    scores = torch.empty((*q.shape[:-1],1024), device=q.device, dtype=torch.float16)
    _lib.causal_scores(qh, kh, scores, threads=(16*256,16,64), group_size=(256,1,1))
    probs = torch.empty_like(scores)
    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*256), group_size=256)
    return (probs @ vh).float()
