"""Fused, tiled causal attention for Apple GPUs."""
import torch


def _source():
    # Generate constant matrix operations so the compiler keeps tiles in registers.
    s = ['''#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void attention(device const float* Q, device const float* K,
                      device const float* V, device float* O,
                      uint2 block [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half keys[4096];
    threadgroup half vals[4096];
    uint base = block.y * 1024 * 64;
    uint qr = block.x * 64 + sg * 8;
    uint r = ((lane >> 1) & 3) + ((lane >> 4) * 4);
    uint c = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
    simdgroup_half8x8 q[8];
    simdgroup_float8x8 out[8];
    float maximum = -INFINITY, denominator = 0.0f;
''']
    for d in range(8):
        s.append(f'''q[{d}].thread_elements()[0] = half(Q[base+(qr+r)*64+{d*8}+c]);
q[{d}].thread_elements()[1] = half(Q[base+(qr+r)*64+{d*8}+c+1]);
out[{d}] = simdgroup_float8x8(0.0f);''')
    s.append('''for (uint start=0; start<=block.x*64; start+=64) {
    for (uint i=tid; i<1024; i+=256) {
        *((threadgroup half4*)keys+i) = half4(*((device const float4*)(K+base+start*64)+i));
        *((threadgroup half4*)vals+i) = half4(*((device const float4*)(V+base+start*64)+i));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 score[8];
    simdgroup_half8x8 p[8];
''')
    for n in range(8):
        s.append(f'score[{n}] = simdgroup_float8x8(0.0f);')
    for d in range(8):
        s.append(f'simdgroup_half8x8 kk{d};')
        for n in range(8):
            s.append(f'''simdgroup_load(kk{d}, keys+{n*8*64+d*8},64,ulong2(0),true);
simdgroup_multiply_accumulate(score[{n}], q[{d}], kk{d}, score[{n}]);''')
    s.append('float mx = -INFINITY;')
    for n in range(8):
        s.append(f'''score[{n}].thread_elements()[0] *= 0.125f;
score[{n}].thread_elements()[1] *= 0.125f;''')
    s.append('if (start == block.x*64) {')
    for n in range(8):
        s.append(f'''
 if (start+{n*8}+c > qr+r) score[{n}].thread_elements()[0] = -INFINITY;
 if (start+{n*8}+c+1 > qr+r) score[{n}].thread_elements()[1] = -INFINITY;''')
    s.append('}')
    for n in range(8):
        s.append(f'mx=max(mx,max(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1]));')
    s.append('''mx=max(mx,simd_shuffle_xor(mx,1));
mx=max(mx,simd_shuffle_xor(mx,8));
float nextmax=max(maximum,mx);
float alpha=fast::exp(maximum-nextmax);
maximum=nextmax;
float total=0.0f;''')
    for n in range(8):
        s.append(f'''float2 prob{n} = fast::exp(float2(score[{n}].thread_elements()[0],score[{n}].thread_elements()[1])-nextmax);
p[{n}].thread_elements()[0]=half(prob{n}.x);
p[{n}].thread_elements()[1]=half(prob{n}.y);
total += prob{n}.x+prob{n}.y;''')
    for d in range(8):
        s.append(f'''out[{d}].thread_elements()[0] *= alpha;
out[{d}].thread_elements()[1] *= alpha;''')
    s.append('''total+=simd_shuffle_xor(total,1);
total+=simd_shuffle_xor(total,8);
denominator=denominator*alpha+total;''')
    for n in range(8):
        s.append(f'simdgroup_half8x8 vv{n};')
        for d in range(8):
            s.append(f'''simdgroup_load(vv{n}, vals+{n*8*64+d*8},64);
simdgroup_multiply_accumulate(out[{d}],p[{n}],vv{n},out[{d}]);''')
    s.append('threadgroup_barrier(mem_flags::mem_threadgroup);\n}')
    for d in range(8):
        s.append(f'''out[{d}].thread_elements()[0] /= denominator;
out[{d}].thread_elements()[1] /= denominator;
simdgroup_store(out[{d}],O+base+qr*64+{d*8},64);''')
    s.append('}')
    return '\n'.join(s)


_lib = torch.mps.compile_shader(_source())


def kernel(q, k, v):
    out = torch.empty_like(q)
    _lib.attention(q, k, v, out, threads=(4096,64,1), group_size=(256,1,1))
    return out
