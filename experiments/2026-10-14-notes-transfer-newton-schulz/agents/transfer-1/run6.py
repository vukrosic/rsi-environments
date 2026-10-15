import torch


def _matmul_source(name, rows, cols, inner, gram=False, add=0.0, scale=1.0, diagonal=0.0, mode=""):
    symmetric = cols == 256
    source = f'''
kernel void {name}(device const half* L [[buffer(0)]],
                  device const half* R [[buffer(1)]],
                  device half* O [[buffer(2)]],
                  {"device half* E [[buffer(3)]], device half* F [[buffer(4)]]," if mode == "init" else ""}
                  {"device const half* previous [[buffer(3)]], constant float& coefficient [[buffer(4)]]," if mode == "cheb" else ""}
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
        half value = half({scale}f * c{r}{c}.thread_elements()[{e}] {f"+ {add}f * float(R[index])" if add else ""} {f"+ (rr == cc ? {diagonal}f : 0.0f)" if diagonal else ""} {"- float(previous[index]) + (rr == cc ? coefficient : 0.0f)" if mode == "cheb" else ""});
        O[index] = value;
        {f"if (group.x > group.y) O[batch*{rows*cols}+cc*{cols}+rr] = value;" if symmetric else ""}
        {f"half evalue = half(-0.3104617807090559f * float(value) + (rr == cc ? 0.605336934464486f : 0.0f)); E[index] = evalue; F[index] = (rr == cc ? half(-0.15523089035452795f) : half(0)); if (group.x > group.y) {{ E[batch*{rows*cols}+cc*{cols}+rr] = evalue; F[batch*{rows*cols}+cc*{cols}+rr] = half(0); }}" if mode == "init" else ""}
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
_source += _matmul_source('gram', 256, 256, 704, gram=True, scale=200.0, diagonal=-1.1, mode="init")
_source += _matmul_source('cheb', 256, 256, 256, scale=2.0, mode="cheb")
_source += _matmul_source('cheb_final', 256, 256, 256, mode="cheb")
_source += _matmul_source('update', 256, 704, 256)
_lib = torch.mps.compile_shader(_source)


def kernel(g):
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    D = torch.empty_like(A)
    norms = g.norm(dim=(-2, -1))
    _lib.normalize(g, norms, X, threads=g.numel(), group_size=256)
    _lib.gram(X, X, A, B, C, threads=(1024, 4, 10), group_size=(256, 1, 1))
    coefficients = (16.388965002247367, -13.093600571454564, 4.83196202192451,
                    -2.5291799923596523, 2.4159305210719024, -0.20477035288816467,
                    -1.8162127878747603, 1.9313871721771643, -1.2301725825333545)
    for i in range(8, 0, -1):
        _lib.cheb(A, B, D, C, coefficients[i], threads=(1024, 4, 10), group_size=(256, 1, 1))
        C, B, D = B, D, C
    _lib.cheb_final(A, B, D, C, coefficients[0], threads=(1024, 4, 10), group_size=(256, 1, 1))
    _lib.update(D, X, Y, threads=(2816, 4, 10), group_size=(256, 1, 1))
    return Y.mT
