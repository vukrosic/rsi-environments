import torch


def _matmul_source(name, rows, cols, inner, gram=False, add=0.0, scale=1.0):
    source = f'''
kernel void {name}(device const half* L [[buffer(0)]],
                  device const half* R [[buffer(1)]],
                  device half* O [[buffer(2)]],
                  uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {{
    threadgroup half left[1024];
    threadgroup half right[1024];
    threadgroup float result[1024];
    uint row = group.y * 32, col = group.x * 32;
    uint batch = group.z;
    uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;
    simdgroup_float8x8 c00(0), c01(0), c10(0), c11(0);
    for (uint kk = 0; kk < {inner}; kk += 32) {{
        for (uint i = tid; i < 1024; i += 128) {{
            uint r = i / 32, c = i % 32;
            left[i] = L[batch * {rows * inner} + (row+r)*{inner}+kk+c];
            right[i] = R[batch * {cols * inner if gram else inner * cols} + {f'(col+c)*{inner}+kk+r' if gram else f'(kk+r)*{cols}+col+c'}];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for k in range(0, 32, 8):
        source += f'''
        {{
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, left + sr*32 + {k}, 32);
            simdgroup_load(a1, left + (sr+8)*32 + {k}, 32);
            simdgroup_load(b0, right + {k}*32 + sc, 32);
            simdgroup_load(b1, right + {k}*32 + sc+8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }}
'''
    source += f'''
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    simdgroup_store(c00, result + sr*32+sc, 32);
    simdgroup_store(c01, result + sr*32+sc+8, 32);
    simdgroup_store(c10, result + (sr+8)*32+sc, 32);
    simdgroup_store(c11, result + (sr+8)*32+sc+8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 128) {{
        uint index = batch*{rows*cols} + (row+i/32)*{cols}+col+i%32;
        O[index] = half({scale}f * result[i] {f'+ {add}f * float(R[index])' if add else ''});
    }}
}}
'''
    return source


_source = '''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void normalize(device const float* G [[buffer(0)]],
                      device const float* norms [[buffer(1)]],
                      device half* X [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    uint b = i / (256*704), j = i % (256*704);
    X[i] = half(G[b*256*704 + (j%704)*256+j/704] / (norms[b]+1e-7f));
}
'''
_source += _matmul_source('gram', 256, 256, 704, gram=True)
_source += _matmul_source('poly', 256, 256, 256, add=-4.775, scale=2.0315)
_source += _matmul_source('update', 256, 704, 256, add=3.4445)
_lib = torch.mps.compile_shader(_source)


def kernel(g):
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    norms = g.norm(dim=(-2, -1))
    _lib.normalize(g, norms, X, threads=g.numel(), group_size=256)
    for _ in range(5):
        _lib.gram(X, X, A, threads=(1024, 8, 10), group_size=(128, 1, 1))
        _lib.poly(A, A, B, threads=(1024, 8, 10), group_size=(128, 1, 1))
        _lib.update(B, X, Y, threads=(2816, 8, 10), group_size=(128, 1, 1))
        X, Y = Y, X
    return X.mT
