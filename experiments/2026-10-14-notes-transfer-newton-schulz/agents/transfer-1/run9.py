import torch


_coefficients = (16.388965002247367, -13.093600571454564, 4.83196202192451,
                 -2.5291799923596523, 2.4159305210719024, -0.20477035288816467,
                 -1.8162127878747603, 1.9313871721771643, -1.2301725825333545,
                 0.605336934464486, -0.15523089035452795)


def _matmul_source(name, rows, cols, inner, gram=False, add=0.0, scale=1.0, diagonal=0.0, mode="", tile=64, basis=0, ktile=64):
    symmetric = cols == 256
    threads = tile * 4
    cmats = tile // 16
    source = f'''
kernel void {name}(device const half* L [[buffer(0)]],
                  device const {"float" if mode == "combine" else "half"}* R [[buffer(1)]],
                  device half* O [[buffer(2)]],
                  {"device float* low [[buffer(3)]], device float* high [[buffer(4)]]," if mode == "init" else ""}
                  {"device const half* Q [[buffer(3)]], device float* low [[buffer(4)]], device float* high [[buffer(5)]]," if mode == "basis" else ""}
                  {"device const float* low [[buffer(3)]]," if mode == "combine" else ""}
                  uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {{
    {"if (group.x < group.y) return;" if symmetric else ""}
    threadgroup half left[{tile*ktile}];
    threadgroup half right[{tile*ktile}];
    uint row = group.y * {tile}, col = group.x * {tile};
    uint batch = group.z;
    uint sr = (sg / 2) * 16, sc = (sg % 2) * {tile//2};
'''
    for r in range(2):
        for c in range(cmats):
            source += f"    simdgroup_float8x8 c{r}{c}(0);\n"
    source += f'''
    for (uint kk = 0; kk < {inner}; kk += {ktile}) {{
        for (uint i = tid; i < {tile*ktile}; i += {threads}) {{
            uint r = i / {ktile}, c = i % {ktile};
            left[i] = L[batch * {rows * inner} + (row+r)*{inner}+kk+c];
            r = i / {tile}; c = i % {tile};
            right[i] = R[batch * {cols * inner if gram else inner * cols} + {f"(col+c)*{inner}+kk+r" if gram else f"(kk+r)*{cols}+col+c"}];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    for k in range(0, ktile, 8):
        source += "        {\n"
        for r in range(2):
            source += f"            simdgroup_half8x8 a{r};\n            simdgroup_load(a{r}, left + (sr+{r*8})*{ktile} + {k}, {ktile});\n"
        for c in range(cmats):
            source += f"            simdgroup_half8x8 b{c};\n            simdgroup_load(b{c}, right + {k}*{tile} + sc+{c*8}, {tile});\n"
        for r in range(2):
            for c in range(cmats):
                source += f"            simdgroup_multiply_accumulate(c{r}{c}, a{r}, b{c}, c{r}{c});\n"
        source += "        }\n"
    source += '''
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint lr = ((lane >> 1) & 3) | ((lane >> 2) & 4);
    uint lc = ((lane << 1) & 2) | ((lane >> 1) & 4);
'''
    for r in range(2):
        for c in range(cmats):
            for e in range(2):
                source += f'''
    {{
        uint rr = row + sr + {r*8} + lr;
        uint cc = col + sc + {c*8+e} + lc;
        uint index = batch*{rows*cols} + rr*{cols}+cc;
        half value = half({scale}f * c{r}{c}.thread_elements()[{e}] {f"+ {add}f * float(R[index])" if add else ""} {f"+ (rr == cc ? {diagonal}f : 0.0f)" if diagonal else ""} {"- float(Q[index])" if mode == "basis" and basis in (3,5) else ""} {"+ low[index]" if mode == "combine" else ""});
        O[index] = value;
        {f"if (group.x > group.y) O[batch*{rows*cols}+cc*{cols}+rr] = value;" if symmetric else ""}
        {f"float lv = {_coefficients[1]-_coefficients[9]}f*float(value) + (rr == cc ? {_coefficients[0]-_coefficients[10]}f : 0.0f); float hv = {2*_coefficients[6]}f*float(value); low[index] = lv; high[index] = hv; if (group.x > group.y) {{ low[batch*{rows*cols}+cc*{cols}+rr] = lv; high[batch*{rows*cols}+cc*{cols}+rr] = hv; }}" if mode == "init" else ""}
        {f"float lv = low[index] + {_coefficients[basis]-(_coefficients[10-basis] if basis<5 else 0)}f*float(value); float hv = high[index] + {2*_coefficients[5+basis]}f*float(value); low[index] = lv; high[index] = hv; if (group.x > group.y) {{ low[batch*{rows*cols}+cc*{cols}+rr] = lv; high[batch*{rows*cols}+cc*{cols}+rr] = hv; }}" if mode == "basis" else ""}
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
for _basis in range(2, 6):
    _source += _matmul_source('basis' + str(_basis), 256, 256, 256, scale=2.0,
                             diagonal=-1.0 if _basis in (2,4) else 0.0,
                             mode="basis", tile=32, basis=_basis)
_source += _matmul_source('combine', 256, 256, 256, mode="combine", tile=32)
_source += _matmul_source('update', 256, 704, 256)
_lib = torch.mps.compile_shader(_source)


def kernel(g):
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    D = torch.empty_like(A)
    low = torch.empty((10, 256, 256), device=g.device, dtype=torch.float32)
    high = torch.empty_like(low)
    norms = g.norm(dim=(-2, -1))
    _lib.normalize(g, norms, X, threads=g.numel(), group_size=256)
    _lib.gram(X, X, A, low, high, threads=(1024, 4, 10), group_size=(256, 1, 1))
    _lib.basis2(A, A, B, A, low, high, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.basis3(A, B, C, A, low, high, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.basis4(B, B, D, A, low, high, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.basis5(B, C, D, A, low, high, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.combine(D, high, B, low, threads=(1024, 8, 10), group_size=(128, 1, 1))
    _lib.update(B, X, Y, threads=(2816, 4, 10), group_size=(256, 1, 1))
    return Y.mT
