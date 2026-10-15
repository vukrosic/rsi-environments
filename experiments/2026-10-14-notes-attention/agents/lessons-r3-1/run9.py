import torch


def _source():
    src = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float* Q [[buffer(0)]],
                      device const float* K [[buffer(1)]],
                      device const float* V [[buffer(2)]],
                      device float* O [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half keys[4096];
    threadgroup half values[4096];
    uint base = group.y * 1024 * 64;
    uint first = group.x * 64;
    uint row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
    uint col = (lane & 1) * 2 + ((lane & 8) >> 1);
    uint qr = first + sg * 8 + row;
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 out[8];
    for (uint z=tid; z<1024; z+=256) {
        ((threadgroup half4*)keys)[z] = half4(((device const float4*)(Q+base+first*64))[z] * 0.125f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    __INIT__
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -INFINITY;
    float denom = 0.0f;
    for (uint start=0; start<=first; start+=64) {
        for (uint z=tid; z<1024; z+=256) {
            ((threadgroup half4*)keys)[z] = half4(((device const float4*)(K+base+start*64))[z]);
            ((threadgroup half4*)values)[z] = half4(((device const float4*)(V+base+start*64))[z]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[8];
        DOT
        float tilemax = -INFINITY;
        SCALE
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
        float nm = max(maximum, tilemax);
        float alpha = fast::exp(maximum - nm);
        float sum = 0.0f;
        simdgroup_half8x8 probs[8];
        PROBS
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        denom = denom * alpha + sum;
        maximum = nm;
        VALUE
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    STORE
}
'''
    src = src.replace('__INIT__', '\n'.join(f'simdgroup_load(qm[{d}], keys+sg*8*64+{d*8}, 64); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
    dot = []
    for j in range(8):
        dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
        for d in range(8):
            dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
    src = src.replace('DOT', '\n'.join(dot))
    scale = ['if (start == first) {']
    for j in range(8):
        for e in range(2):
            scale.append(f'if (start+{j*8}+col+{e} > qr) scores[{j}].thread_elements()[{e}] = -INFINITY;')
    scale.append('}')
    for j in range(8):
        for e in range(2):
            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}]; tilemax = max(tilemax,x); }}')
    src = src.replace('SCALE', '\n'.join(scale))
    probs = []
    for j in range(8):
        for e in range(2):
            probs.append(f'{{ float p = fast::exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
    src = src.replace('PROBS', '\n'.join(probs))
    vals = []
    for d in range(8):
        vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
        for j in range(0, 8, 2):
            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
            for z in range(j, j + 2):
                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
    src = src.replace('VALUE', '\n'.join(vals))
    src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
    return src


_shader = torch.mps.compile_shader(_source())


def kernel(q, k, v):
    out = torch.empty_like(q)
    _shader.attention(q, k, v, out, threads=(16*256, 64, 1), group_size=(256, 1, 1))
    return out
