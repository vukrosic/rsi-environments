import torch


def _matmul_source(name, rows, cols, inner, gram=False, add=0.0, scale=1.0, diagonal=0.0):
    symmetric = cols == 256
    source = f'''
kernel void {name}(device const half* L [[buffer(0)]],
                  device const half* R [[buffer(1)]],
                  device half* O [[buffer(2)]],
                  uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {{
    {"if (group.x < group.y) return;" if symmetric else ""}
    threadgroup half left[2048];
    threadgroup half right[2048];
    uint row = group.y * 64, col = group.x * 64;
    uint batch = group.z;
    uint sr = (sg / 2) * 16, sc = (sg % 2) * 32;
'''
    for r in range(2):
        for c in range(4):
            source += f"    simdgroup_float8x8 c{r}{c}(0);\n"
    source += f'''
    for (uint kk = 0; kk < {inner}; kk += 32) {{
        for (uint i = tid; i < 2048; i += 256) {{
            uint r = i / 32, c = i % 32;
            left[i] = L[batch * {rows * inner} + (row+r)*{inner}+kk+c];
            r = i / 64; c = i % 64;
            right[i] = R[batch * {cols * inner if gram else inner * cols} + {f"(col+c)*{inner}+kk+r" if gram else f"(kk+r)*{cols}+col+c"}];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for k in range(0, 32, 8):
        source += "        {\n"
        for r in range(2):
            source += f"            simdgroup_half8x8 a{r};\n            simdgroup_load(a{r}, left + (sr+{r*8})*32 + {k}, 32);\n"
        for c in range(4):
            source += f"            simdgroup_half8x8 b{c};\n            simdgroup_load(b{c}, right + {k}*64 + sc+{c*8}, 64);\n"
        for r in range(2):
            for c in range(4):
                source += f"            simdgroup_multiply_accumulate(c{r}{c}, a{r}, b{c}, c{r}{c});\n"
        source += "        }\n"
    source += '''
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint lr = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint lc = ((lane << 1) & 2) | ((lane >> 1) & 4);
'''
    for r in range(2):
        for c in range(4):
            for e in range(2):
                source += f'''
    {{
        uint rr = row + sr + {r*8} + lr;
        uint cc = col + sc + {c*8+e} + lc;
        uint index = batch*{rows*cols} + rr*{cols}+cc;
        half value = half({scale}f * c{r}{c}.thread_elements()[{e}] {f"+ {add}f * float(R[index])" if add else ""} {f"+ (rr == cc ? {diagonal}f : 0.0f)" if diagonal else ""});
        O[index] = value;
        {f"if (group.x > group.y) O[batch*{rows*cols}+cc*{cols}+rr] = value;" if symmetric else ""}
    }}
'''
    return source + "}\n"


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
_source += _matmul_source('poly', 256, 256, 256, add=-4.775, scale=2.0315, diagonal=3.4445)
_source += _matmul_source('propagate', 256, 256, 256)
_source += _matmul_source('update', 256, 704, 256)
_lib = torch.mps.compile_shader(_source)


def kernel(g):
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    T = torch.empty_like(A)
    U = torch.empty_like(A)
    norms = g.norm(dim=(-2, -1))
    _lib.normalize(g, norms, X, threads=g.numel(), group_size=256)
    _lib.gram(X, X, A, threads=(1024, 4, 10), group_size=(256, 1, 1))
    for i in range(5):
        if i == 0:
            _lib.poly(A, A, T, threads=(1024, 4, 10), group_size=(256, 1, 1))
            P = T
        else:
            _lib.poly(A, A, B, threads=(1024, 4, 10), group_size=(256, 1, 1))
            _lib.propagate(B, T, U, threads=(1024, 4, 10), group_size=(256, 1, 1))
            T, U = U, T
            P = B
        if i < 4:
            _lib.propagate(A, P, C, threads=(1024, 4, 10), group_size=(256, 1, 1))
            _lib.propagate(P, C, A, threads=(1024, 4, 10), group_size=(256, 1, 1))
    _lib.update(T, X, Y, threads=(2816, 4, 10), group_size=(256, 1, 1))
    return Y.mT
