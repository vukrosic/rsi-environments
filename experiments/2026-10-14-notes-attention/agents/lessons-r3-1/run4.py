import torch


_QSIZE = 64
_KSIZE = 32


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
    threadgroup half keys[__KEY_STORAGE__];
    threadgroup half values[__KEY_STORAGE__];
    uint base = group.y * 1024 * 64;
    uint first = group.x * __QSIZE__;
    uint row = ((lane & 7) >> 1) + ((lane >> 4) << 2);
    uint col = (lane & 1) * 2 + ((lane & 8) >> 1);
    uint qr = first + sg * 8 + row;
    simdgroup_half8x8 qm[8];
    simdgroup_float8x8 out[8];
    __INIT__
    float maximum = -INFINITY;
    float denom = 0.0f;
    for (uint start=0; start<first+__QSIZE__; start+=__KSIZE__) {
        for (uint z=tid; z<__KEY_VECTORS__; z+=__THREADS__) {
            ((threadgroup half4*)keys)[z] = half4(((device const float4*)(K+base+start*64))[z]);
            ((threadgroup half4*)values)[z] = half4(((device const float4*)(V+base+start*64))[z]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores[__KMATS__];
        DOT
        float tilemax = -INFINITY;
        SCALE
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
        float nm = max(maximum, tilemax);
        float alpha = exp(maximum - nm);
        float sum = 0.0f;
        simdgroup_half8x8 probs[__KMATS__];
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
    src = src.replace('__INIT__', '\n'.join(f'qm[{d}].thread_elements()[0] = half(Q[base+qr*64+{d*8}+col]); qm[{d}].thread_elements()[1] = half(Q[base+qr*64+{d*8}+col+1]); out[{d}] = simdgroup_float8x8(0.0f);' for d in range(8)))
    dot = []
    for j in range(_KSIZE // 8):
        dot.append(f'scores[{j}] = simdgroup_float8x8(0.0f);')
        for d in range(8):
            dot.append(f'{{ simdgroup_half8x8 km; simdgroup_load(km, keys+{j*8*64+d*8}, 64, ulong2(0), true); simdgroup_multiply_accumulate(scores[{j}], qm[{d}], km, scores[{j}]); }}')
    src = src.replace('DOT', '\n'.join(dot))
    scale = []
    for j in range(_KSIZE // 8):
        for e in range(2):
            scale.append(f'{{ float x = scores[{j}].thread_elements()[{e}] * 0.125f; if (start + {_KSIZE} > first && start+{j*8}+col+{e} > qr) x = -INFINITY; scores[{j}].thread_elements()[{e}] = x; tilemax = max(tilemax,x); }}')
    src = src.replace('SCALE', '\n'.join(scale))
    probs = []
    for j in range(_KSIZE // 8):
        for e in range(2):
            probs.append(f'{{ float p = exp(scores[{j}].thread_elements()[{e}] - nm); sum += p; probs[{j}].thread_elements()[{e}] = half(p); }}')
    src = src.replace('PROBS', '\n'.join(probs))
    vals = []
    for d in range(8):
        vals.append(f'out[{d}].thread_elements()[0] *= alpha; out[{d}].thread_elements()[1] *= alpha;')
        for j in range(0, _KSIZE // 8, 2):
            vals.append('{ simdgroup_half8x8 partial(half(0.0f));')
            for z in range(j, j + 2):
                vals.append(f'{{ simdgroup_half8x8 vm; simdgroup_load(vm, values+{z*8*64+d*8},64); simdgroup_multiply_accumulate(partial, probs[{z}], vm, partial); }}')
            vals.append(f'out[{d}].thread_elements()[0] += float(partial.thread_elements()[0]); out[{d}].thread_elements()[1] += float(partial.thread_elements()[1]); }}')
    src = src.replace('VALUE', '\n'.join(vals))
    src = src.replace('STORE', '\n'.join(f'O[base+qr*64+{d*8}+col+{e}] = out[{d}].thread_elements()[{e}] / denom;' for d in range(8) for e in range(2)))
    return (src.replace('__QSIZE__', str(_QSIZE))
            .replace('__KSIZE__', str(_KSIZE))
            .replace('__KEY_STORAGE__', str(_KSIZE * 64))
            .replace('__QUERY_VECTORS__', str(_QSIZE * 16))
            .replace('__KEY_VECTORS__', str(_KSIZE * 16))
            .replace('__THREADS__', str(_QSIZE * 4))
            .replace('__KMATS__', str(_KSIZE // 8)))


_shader = torch.mps.compile_shader(_source())


def kernel(q, k, v):
    out = torch.empty_like(q)
    _shader.attention(q, k, v, out, threads=(1024 // _QSIZE * (_QSIZE * 4), 64, 1), group_size=(_QSIZE * 4, 1, 1))
    return out
