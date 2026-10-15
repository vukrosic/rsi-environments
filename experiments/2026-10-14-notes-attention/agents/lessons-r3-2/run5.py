"""Fused causal attention for the M4 GPU."""
import torch

_QROWS = 128
_KCOLS = 64

def _source():
    n = _KCOLS // 8
    code = r'''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *k [[buffer(1)]],
                      device const float *v [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint qb = group.x * QROWS;
    const uint base = group.y * 1024 * 64;
    const uint row = ((lane >> 1) & 3) + ((lane >> 2) & 4);
    const uint col = (lane & 1) * 2 + (lane & 8) / 2;
    const uint qr = qb + sg * 8 + row;
    threadgroup half keys[KCOLS * 64];
    threadgroup half vals[KCOLS * 64];
    simdgroup_half8x8 Q[8], P[KCOLS / 8];
    simdgroup_float8x8 S[KCOLS / 8], O[8];
    float maximum = -INFINITY, denom = 0.0f;
'''
    for d in range(8):
        code += f'''    {{ float2 x = *((device const float2 *)(q + base + qr * 64 + {d*8} + col));
    Q[{d}].thread_elements()[0] = half(x.x);
    Q[{d}].thread_elements()[1] = half(x.y); }}
    O[{d}] = simdgroup_float8x8(0.0f);
'''
    code += '''    for (uint kb = 0; kb < qb + QROWS; kb += KCOLS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KCOLS * 16; i += QROWS * 4) {
            *((threadgroup half4 *)(keys + i * 4)) = half4(*((device const float4 *)(k + base + kb * 64 + i * 4)));
            *((threadgroup half4 *)(vals + i * 4)) = half4(*((device const float4 *)(v + base + kb * 64 + i * 4)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for j in range(n):
        code += f'        S[{j}] = simdgroup_float8x8(0.0f);\n'
    for d in range(8):
        code += '        { simdgroup_half8x8 K;\n'
        for j in range(n):
            code += f'''        simdgroup_load(K, keys + {j*8*64+d*8}, 64, ulong2(0), true);
        simdgroup_multiply_accumulate(S[{j}], Q[{d}], K, S[{j}]);
'''
        code += '        }\n'
    code += '        float tilemax = -INFINITY;\n'
    for j in range(n):
        code += f'''        S[{j}].thread_elements()[0] *= 0.18033688011112042f;
        S[{j}].thread_elements()[1] *= 0.18033688011112042f;
        if (kb + KCOLS > qb) {{
            if (kb + {j*8} + col > qr) S[{j}].thread_elements()[0] = -INFINITY;
            if (kb + {j*8} + col + 1 > qr) S[{j}].thread_elements()[1] = -INFINITY;
        }}
        tilemax = max(tilemax, max(S[{j}].thread_elements()[0], S[{j}].thread_elements()[1]));
'''
    code += '''        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1));
        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8));
        float newmax = max(maximum, tilemax);
        float correction = fast::exp2(maximum - newmax);
        float tilesum = 0.0f;
'''
    for j in range(n):
        code += f'''        {{ float a = fast::exp2(S[{j}].thread_elements()[0] - newmax);
        float b = fast::exp2(S[{j}].thread_elements()[1] - newmax);
        tilesum += a + b;
        P[{j}].thread_elements()[0] = half(a);
        P[{j}].thread_elements()[1] = half(b); }}
'''
    code += '''        tilesum += simd_shuffle_xor(tilesum, 1);
        tilesum += simd_shuffle_xor(tilesum, 8);
        denom = denom * correction + tilesum;
        maximum = newmax;
'''
    for d in range(8):
        code += f'''        O[{d}].thread_elements()[0] *= correction;
        O[{d}].thread_elements()[1] *= correction;
'''
    for j in range(n):
        code += '        { simdgroup_half8x8 V;\n'
        for d in range(8):
            code += f'''        simdgroup_load(V, vals + {j*8*64+d*8}, 64);
        simdgroup_multiply_accumulate(O[{d}], P[{j}], V, O[{d}]);
'''
        code += '        }\n'
    code += '    }\n'
    for d in range(8):
        code += f'''    *((device float2 *)(out + base + qr * 64 + {d*8} + col)) = float2(O[{d}].thread_elements()[0], O[{d}].thread_elements()[1]) / denom;
'''
    return (code + '}\n').replace('QROWS', str(_QROWS)).replace('KCOLS', str(_KCOLS))


_lib = torch.mps.compile_shader(_source())


def kernel(q, k, v):
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out, threads=(4096, 64, 1), group_size=(_QROWS * 4, 1, 1))
    return out
