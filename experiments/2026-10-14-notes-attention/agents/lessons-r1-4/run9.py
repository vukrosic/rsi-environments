"""Causal attention with half precision matrix products and fused softmax."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
kernel void causal_softmax(device const half* scores [[buffer(0)]],
                           device half* probs [[buffer(1)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    uint base = row * 4 * 1024 + sg * 1024;
    uint pos = (row * 4 + sg) % 1024;
    float x[32];
    float mx = -INFINITY;
    for (uint i=0; i<32; ++i) {
        uint col = lane + i*32;
        x[i] = col <= pos ? float(scores[base+col])*0.125f : -INFINITY;
        mx = max(mx,x[i]);
    }
    mx = simd_max(mx);
    float sum = 0;
    for (uint i=0; i<32; ++i) {
        x[i] = fast::exp(x[i]-mx);
        sum += x[i];
    }
    sum = simd_sum(sum);
    float inv = 1.0f/sum;
    for (uint i=0; i<32; ++i) probs[base+lane+i*32]=half(x[i]*inv);
}
""")

def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    scores = qh @ kh.transpose(-1,-2)
    probs = torch.empty_like(scores)
    _lib.causal_softmax(scores, probs, threads=(q.numel()//64*32), group_size=128)
    return (probs @ vh).float()
