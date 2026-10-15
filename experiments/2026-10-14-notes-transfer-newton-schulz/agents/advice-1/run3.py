"""Batched Newton-Schulz orthogonalization on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void mm(device const half* a, device const half* b, device half* out,
               constant uint& M, constant uint& N, constant uint& K,
               constant uint& transpose_b, constant uint& mode,
               uint3 group [[threadgroup_position_in_grid]],
               uint tid [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half at[1024];
    threadgroup half bt[1024];
    threadgroup float ct[1024];
    uint row = group.y * 32;
    uint col = group.x * 32;
    uint batch = group.z;
    a += batch * M * K;
    b += batch * K * N;
    out += batch * M * N;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;
    for (uint k = 0; k < K; k += 32) {
        for (uint q = tid; q < 1024; q += 128) {
            uint r = q / 32, c = q % 32;
            at[q] = a[(row + r) * K + k + c];
            bt[q] = transpose_b ? b[(col + c) * K + k + r] : b[(k + r) * N + col + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, at + sr * 32 + kk, 32);
            simdgroup_load(a1, at + (sr + 8) * 32 + kk, 32);
            simdgroup_load(b0, bt + kk * 32 + sc, 32);
            simdgroup_load(b1, bt + kk * 32 + sc + 8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, ct + sr * 32 + sc, 32);
    simdgroup_store(c01, ct + sr * 32 + sc + 8, 32);
    simdgroup_store(c10, ct + (sr + 8) * 32 + sc, 32);
    simdgroup_store(c11, ct + (sr + 8) * 32 + sc + 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < 1024; q += 128) {
        uint r = row + q / 32, c = col + q % 32;
        float v = ct[q];
        if (mode == 1) v = 2.0315f * v - 4.7750f * float(a[r*K+c]) + (r==c ? 3.4445f : 0.0f);
        out[r*N+c] = half(v);
    }
}
""")

def _mm(a, b, transpose_b=False, mode=0):
    M, K = a.shape[-2:]
    N = b.shape[-2] if transpose_b else b.shape[-1]
    out = torch.empty((a.shape[0], M, N), dtype=a.dtype, device=a.device)
    _lib.mm(a, b, out, M, N, K, int(transpose_b), mode,
            threads=(N//32*128, M//32, a.shape[0]), group_size=(128,1,1))
    return out


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half().contiguous()
    A = _mm(X, X, True)
    P = None
    for i in range(5):
        T = _mm(A, A, mode=1)
        P = T if i == 0 else _mm(T, P)
        if i != 4:
            A = _mm(_mm(T, A), T, True)
    X = _mm(P, X)
    return X.mT if transposed else X
