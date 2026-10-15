import re
import torch


def _source():
    source = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float4* Q [[buffer(0)]],
                      device const float4* K [[buffer(1)]],
                      device const float4* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint base = group.y * 1024 * 64;
    const uint qs = group.x * 64;
    const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
    const uint col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
    const uint qr = qs + sg * 8 + row;
    threadgroup half keys[4096];
    threadgroup half vals[4096];
    for (uint i = tid; i < 1024; i += 256) {
        ((threadgroup half4*)keys)[i] = half4(Q[base/4 + qs*16 + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 q[8];
    simdgroup_float8x8 out[8];
    #pragma unroll
    for (uint d=0; d<8; ++d) {
        simdgroup_load(q[d], keys + sg*8*64 + d*8, 64);
        out[d] = simdgroup_float8x8(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float running_max = -INFINITY;
    float denominator = 0.0f;
    for (uint ks=0; ks<=qs; ks+=64) {
        for (uint i=tid; i<1024; i+=256) {
            ((threadgroup half4*)keys)[i] = half4(K[base/4 + ks*16 + i]);
            ((threadgroup half4*)vals)[i] = half4(V[base/4 + ks*16 + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[8];
        #pragma unroll
        for (uint j=0; j<8; ++j) { scores[j] = simdgroup_float8x8(0.0f); }
        #pragma unroll
        for (uint d=0; d<8; ++d) {
            #pragma unroll
            for (uint j=0; j<8; ++j) {
                simdgroup_half8x8 km;
                simdgroup_load(km, keys + j*8*64 + d*8, 64, ulong2(0), true);
                simdgroup_multiply_accumulate(scores[j], q[d], km, scores[j]);
            }
        }
        float mx = -INFINITY;
        #pragma unroll
        for (uint j=0; j<8; ++j) {
            #pragma unroll
            for (uint e=0; e<2; ++e) {
                float s = scores[j].thread_elements()[e] * 0.18033688011112042f;
                if (ks==qs && ks+j*8+col+e>qr) s=-INFINITY;
                scores[j].thread_elements()[e]=s;
                mx=max(mx,s);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx,1));
        mx = max(mx, simd_shuffle_xor(mx,8));
        float nm = max(running_max,mx);
        float alpha = fast::exp2(running_max-nm);
        float sum = 0.0f;
        simdgroup_half8x8 p[8];
        #pragma unroll
        for (uint j=0; j<8; ++j) {
            #pragma unroll
            for (uint e=0; e<2; ++e) {
                float z=fast::exp2(scores[j].thread_elements()[e]-nm);
                p[j].thread_elements()[e]=half(z);
                sum+=z;
            }
        }
        sum += simd_shuffle_xor(sum,1);
        sum += simd_shuffle_xor(sum,8);
        denominator=denominator*alpha+sum;
        running_max=nm;
        #pragma unroll
        for (uint d=0; d<8; ++d) {
            out[d].thread_elements()[0]*=alpha;
            out[d].thread_elements()[1]*=alpha;
            #pragma unroll
            for (uint j=0; j<8; ++j) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vals+j*8*64+d*8,64);
                simdgroup_multiply_accumulate(out[d],p[j],vm,out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint d=0; d<8; ++d) {
        O[base+qr*64+d*8+col]=out[d].thread_elements()[0]/denominator;
        O[base+qr*64+d*8+col+1]=out[d].thread_elements()[1]/denominator;
    }
}
'''
    pattern = re.compile(r'#pragma unroll\s+for \(uint (\w+)=(\d+); \1<(\d+); \+\+\1\) \{')
    while matches := list(pattern.finditer(source)):
        match = matches[-1]
        end = match.end()
        depth = 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[match.end():end-1]
        expanded = '\n'.join('{' + re.sub(r'\b'+match[1]+r'\b', str(i), body) + '}'
                             for i in range(int(match[2]), int(match[3])))
        source = source[:match.start()] + expanded + source[end:]
    return source


_lib = torch.mps.compile_shader(_source())


def kernel(q, k, v):
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(256, 1, 1))
    return out
