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
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qt[32*64];
    threadgroup half kv[64*64];
    threadgroup float score[32*64];
    threadgroup half weight[32*64];
    threadgroup float denom[32];
    uint base = gid.y*1024*64;
    uint row = gid.x*32;
    for (uint i=tid;i<32*64;i+=128) qt[i]=q[base+row*64+i];
    if(tid<32) denom[tid]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 output[8];
    for(uint j=0;j<8;j++) output[j]=simdgroup_float8x8(0.0f);
    for(uint start=0;start<row+32;start+=64) {
        for(uint i=tid;i<64*64;i+=128) kv[i]=k[base+start*64+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[8];
        for(uint j=0;j<8;j++) scores[j]=simdgroup_float8x8(0.0f);
        for(uint d=0;d<8;d++) {
            simdgroup_half8x8 a;
            simdgroup_load(a,qt+sg*8*64+d*8,64);
            for(uint j=0;j<8;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,kv+j*8*64+d*8,64,ulong2(0),true);
                simdgroup_multiply_accumulate(scores[j],a,b,scores[j]);
            }
        }
        for(uint j=0;j<8;j++) simdgroup_store(scores[j],score+sg*8*64+j*8,64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32) {
            float sum=0.0f;
            for(uint j=0;j<64;j++) {
                half w = start+j<=row+tid ? half(fast::exp(score[tid*64+j]*0.125f)) : half(0.0h);
                weight[tid*64+j]=w;
                sum+=float(w);
            }
            denom[tid]+=sum;
        }
        for(uint i=tid;i<64*64;i+=128) kv[i]=v[base+start*64+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d=0;d<8;d++) {
            simdgroup_half8x8 a;
            simdgroup_load(a,weight+sg*8*64+d*8,64);
            for(uint j=0;j<8;j++) {
                simdgroup_half8x8 b;
                simdgroup_load(b,kv+d*8*64+j*8,64);
                simdgroup_multiply_accumulate(output[j],a,b,output[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<8;j++) simdgroup_store(output[j],score+sg*8*64+j*8,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=128) out[base+row*64+i]=score[i]/denom[i/64];
}
'''
_lib = torch.mps.compile_shader(_source)

def kernel(q, k, v):
    if q.shape != (8, 8, 1024, 64):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = torch.empty_like(q)
    _lib.attention(q.half(), k.half(), v.half(), out,
                   threads=(32*128,64,1), group_size=(128,1,1))
    return out
