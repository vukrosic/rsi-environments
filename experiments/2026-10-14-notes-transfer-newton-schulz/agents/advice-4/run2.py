import torch


def _matmul(name, n, k, gram=False, update=False):
    rhs_load = ('R[(batch * 256 + col + r) * 704 + kk + c]' if gram
                else f'R[(batch * {k} + kk + r) * {n} + col + c]')
    rhs_store = 'right[c * 32 + r]' if gram else 'right[t]'
    epilogue = ('v' if gram else
                ('3.4445f * float(R[(batch * 256 + row + r) * 704 + col + c]) + v'
                 if update else
                 '2.0315f * v - 4.7750f * float(L[(batch * 256 + row + r) * 256 + col + c])'))
    return f'''
kernel void {name}(device const half* L, device const half* R,
                  device half* O, uint3 group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {{
    const uint row = group.y * 32, col = group.x * 32, batch = group.z;
    threadgroup half left[1024], right[1024];
    threadgroup float result[1024];
    simdgroup_float8x8 c00(0), c01(0), c10(0), c11(0);
    for (uint kk = 0; kk < {k}; kk += 32) {{
        for (uint t = tid; t < 1024; t += 128) {{
            uint r = t / 32, c = t % 32;
            left[t] = L[(batch * 256 + row + r) * {k} + kk + c];
            {rhs_store} = {rhs_load};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < 32; s += 8) {{
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, left + (sg / 2 * 16) * 32 + s, 32);
            simdgroup_load(a1, left + (sg / 2 * 16 + 8) * 32 + s, 32);
            simdgroup_load(b0, right + s * 32 + sg % 2 * 16, 32);
            simdgroup_load(b1, right + s * 32 + sg % 2 * 16 + 8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    simdgroup_store(c00, result + sg / 2 * 16 * 32 + sg % 2 * 16, 32);
    simdgroup_store(c01, result + sg / 2 * 16 * 32 + sg % 2 * 16 + 8, 32);
    simdgroup_store(c10, result + (sg / 2 * 16 + 8) * 32 + sg % 2 * 16, 32);
    simdgroup_store(c11, result + (sg / 2 * 16 + 8) * 32 + sg % 2 * 16 + 8, 32);
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
'''
_lib = torch.mps.compile_shader(_source + _matmul('gram', 256, 704, gram=True)
                               + _matmul('square', 256, 256)
                               + _matmul('update', 704, 256, update=True))


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
    Y = torch.empty_like(X)
    A = torch.empty((10, 256, 256), device=g.device, dtype=torch.float16)
    B = torch.empty_like(A)
    _lib.normalize(g, norms, X, threads=(1024, 22, 10), group_size=(128, 1, 1))
    for _ in range(5):
        _lib.gram(X, X, A, threads=(1024, 8, 10), group_size=(128, 1, 1))
        _lib.square(A, A, B, threads=(1024, 8, 10), group_size=(128, 1, 1))
        _lib.update(B, X, Y, threads=(2816, 8, 10), group_size=(128, 1, 1))
        X, Y = Y, X
    return X.mT
