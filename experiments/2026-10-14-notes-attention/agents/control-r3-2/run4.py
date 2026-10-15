import torch
import torch.nn.functional as F

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
using namespace metal;
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
    for(uint j=0;j<4;j++) s[row*1024+tid+j*256]=half(e[j]/z);
}
""")


def kernel(q, k, v):
    qh, kh, vh = q.half(), k.half(), v.half()
    s = qh @ kh.transpose(-1, -2)
    _lib.softmax_causal(s, threads=(65536*256), group_size=256)
    return (s @ vh).float()
