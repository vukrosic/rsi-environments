import torch


def _matmul(name, n, k, gram=False, update=False):
    tile = 32
    width = 2
    declarations = '\n'.join(f'simdgroup_float8x8 c{i}{j}(0);' for i in range(width) for j in range(width))
    loads = '\n'.join(
        f'simdgroup_half8x8 a{i}, b{i};\n'
        f'simdgroup_load(a{i}, left + (sg / 2 * 16 + {i * 8}) * 32 + s, 32);\n'
        f'simdgroup_load(b{i}, right + s * 32 + sg % 2 * 16 + {i * 8}, 32);'
        for i in range(width))
    products = '\n'.join(f'simdgroup_multiply_accumulate(c{i}{j}, a{i}, b{j}, c{i}{j});'
                         for i in range(width) for j in range(width))
    stores = '\n'.join(f'simdgroup_store(c{i}{j}, result + (sg / 2 * 16 + {i * 8}) * 32 + sg % 2 * 16 + {j * 8}, 32);'
                       for i in range(width) for j in range(width))
    rhs_load = ('R[(batch * 256 + col + r) * 704 + kk + c]' if gram
                else f'R[(batch * {k} + kk + r) * {n} + col + c]')
    rhs_store = 'right[c * 32 + r]' if gram else 'right[t]'
    rhs_indices = 'uint r = t / 32, c = t % 32;'
    epilogue = ('2.0315f * v - 4.7750f * float(L[(batch * 256 + row + r) * 256 + col + c]) + ((row + r == col + c) ? 3.4445f : 0.0f)' if update else 'v')
    return f'''
kernel void {name}(device const half* L, device const half* R,
                  device half* O, uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {{
    const uint row = group.y * 32, col = group.x * 32, batch = group.z;
    threadgroup half left[1024], right[1024];
    threadgroup float result[1024];
    {declarations}
    for (uint kk = 0; kk < {k}; kk += 32) {{
        for (uint t = tid; t < 1024; t += 128) {{
            uint r = t / 32, c = t % 32;
            left[t] = L[(batch * 256 + row + r) * {k} + kk + c];
        }}
        for (uint t = tid; t < 1024; t += 128) {{
            {rhs_indices}
            {rhs_store} = {rhs_load};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < 32; s += 8) {{
            {loads}
            {products}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    {stores}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < 1024; t += 128) {{
        uint r = t / 32, c = t % 32;
        float v = result[t];
        O[(batch * 256 + row + r) * {n} + col + c] = half({epilogue});
    }}
}}
'''


_source = '''
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void normalize(device const float* G, device const float* norms,
                      device half* X, uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[1024];
    uint batch = group.z, row = group.y * 32, col = group.x * 32;
    float scale = 1.0f / (norms[batch] + 1e-7f);
    for (uint t = tid; t < 1024; t += 128) {
        uint r = t / 32, c = t % 32;
        tile[c * 32 + r] = half(G[(batch * 704 + row + r) * 256 + col + c] * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < 1024; t += 128) {
        uint r = t / 32, c = t % 32;
        X[(batch * 256 + col + r) * 704 + row + c] = tile[t];
    }
}
kernel void combine_b(device const half* A, device const half* AA,
                      device half* B, uint t [[thread_position_in_grid]]) {
    B[t] = half(2.0315f * float(AA[t]) - 4.7750f * float(A[t]));
}
kernel void combine_x(device const half* X, device const half* BX,
                      device half* Y, uint t [[thread_position_in_grid]]) {
    Y[t] = half(3.4445f * float(X[t]) + float(BX[t]));
}
kernel void polynomial(device const half* A, device const half* AA,
                      device half* P, uint t [[thread_position_in_grid]]) {
    float v = 2.0315f * float(AA[t]) - 4.7750f * float(A[t]);
    if ((t % 65536) / 256 == t % 256) v += 3.4445f;
    P[t] = half(v);
}
kernel void polynomial_pack(device const half* A, device const half* AA,
                           device const half* S, device half* P, device half* Z,
                           uint t [[thread_position_in_grid]]) {
    float v = 2.0315f * float(AA[t]) - 4.7750f * float(A[t]);
    if ((t % 65536) / 256 == t % 256) v += 3.4445f;
    P[t] = half(v);
    uint z = (t / 256) * 512 + t % 256;
    Z[z] = A[t];
    Z[z + 256] = S[t];
}
'''
_lib = torch.mps.compile_shader(_source + _matmul('square', 256, 256, update=True))


def kernel(g):
    if tuple(g.shape) != (10, 704, 256):
        X = g.mT if g.size(-2) > g.size(-1) else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            A = X @ X.mT
            X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
        return X.mT if g.size(-2) > g.size(-1) else X
    norms = g.norm(dim=(-2, -1))
    X = torch.empty((10, 256, 704), device=g.device, dtype=torch.float16)
    _lib.normalize(g, norms, X, threads=(1024, 22, 10), group_size=(128, 1, 1))
    A = torch.bmm(X, X.mT)
    S = None
    for iteration in range(5):
        P = torch.empty_like(A)
        _lib.square(A, A, P, threads=(1024, 8, 10), group_size=(128, 1, 1))
        S = P if S is None else torch.bmm(P, S)
        if iteration < 4:
            AP = torch.bmm(A, P)
            A = torch.bmm(P, AP)
    return torch.bmm(S, X).mT
