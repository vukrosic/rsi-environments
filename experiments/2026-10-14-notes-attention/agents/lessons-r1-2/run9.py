import torch
import torch.nn.functional as F

_source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void attention(device const half* Q, device const half* K,
                      device const half* V, device float* O,
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint start = group.x * 64;
    const uint base = group.y * 1024 * 64;
    Q += base; K += base; V += base; O += base;
    threadgroup half scores[64*64];
    threadgroup half* probs = scores;
    threadgroup float oldscale[64];
    threadgroup float maxima[64];
    threadgroup float sums[64];
    simdgroup_float8x8 results[8];
    for(uint i=0;i<8;++i) results[i]=simdgroup_float8x8(0.0f);
    const uint row=(lane/2)%4+(lane/16)*4;
    if (tid<64) { maxima[tid]=-INFINITY; sums[tid]=0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key=0; key<start+64; key+=64) {
        for (uint rt=0; rt<4; ++rt) {
          for (uint ct=0; ct<2; ++ct) {
            uint r=(sg/4+rt*2)*8, c=(sg%4+ct*4)*8;
            simdgroup_float8x8 acc(0.0f);
            for(uint d=0;d<64;d+=8) {
                simdgroup_half8x8 a,b;
                simdgroup_load(a,Q+(start+r)*64+d,64);
                simdgroup_load(b,K+(key+c)*64+d,64,ulong2(0),true);
                simdgroup_multiply_accumulate(acc,a,b,acc);
            }
            simdgroup_half8x8 hs;
            hs.thread_elements()[0]=half(acc.thread_elements()[0]);
            hs.thread_elements()[1]=half(acc.thread_elements()[1]);
            simdgroup_store(hs,scores+r*64+c,64);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint ri=0;ri<8;++ri) {
            uint r=sg*8+ri;
            float s=key+lane<=start+r ? float(scores[r*64+lane])*0.125f : -INFINITY;
            float s2=key+lane+32<=start+r ? float(scores[r*64+lane+32])*0.125f : -INFINITY;
            float m=max(maxima[r],simd_max(max(s,s2)));
            float f=fast::exp(maxima[r]-m);
            float p=fast::exp(s-m);
            float p2=fast::exp(s2-m);
            float z=simd_sum(p+p2);
            probs[r*64+lane]=half(p);
            probs[r*64+lane+32]=half(p2);
            if(lane==0) { maxima[r]=m; sums[r]=sums[r]*f+z; oldscale[r]=f; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint rt=0;rt<4;++rt) {
            uint r=(sg/4+rt*2)*8;
            for(uint ct=0;ct<2;++ct) {
                uint c=(sg%4+ct*4)*8;
                uint ix=rt*2+ct;
                results[ix].thread_elements()[0]*=oldscale[r+row];
                results[ix].thread_elements()[1]*=oldscale[r+row];
                for(uint d=0;d<64;d+=8) {
                    simdgroup_half8x8 a,b;
                    simdgroup_load(a,probs+r*64+d,64);
                    simdgroup_load(b,V+(key+d)*64+c,64);
                    simdgroup_multiply_accumulate(results[ix],a,b,results[ix]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint rt=0;rt<4;++rt) {
        uint r=(sg/4+rt*2)*8;
        for(uint ct=0;ct<2;++ct) {
            uint c=(sg%4+ct*4)*8, ix=rt*2+ct;
            results[ix].thread_elements()[0]/=sums[r+row];
            results[ix].thread_elements()[1]/=sums[r+row];
            simdgroup_store(results[ix],O+(start+r)*64+c,64);
        }
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
