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
    threadgroup half at[2048];
    threadgroup half bt[2048];
    threadgroup float ct[4096];
    uint row = group.y * 64;
    uint col = group.x * 64;
    uint batch = group.z;
    a += batch * M * K;
    b += batch * K * N;
    out += batch * M * N;
    simdgroup_float8x8 acc[16];
    for (uint u=0; u<16; ++u) acc[u] = simdgroup_float8x8(0.0f);
    uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    for (uint k = 0; k < K; k += 32) {
        for (uint q = tid; q < 2048; q += 128) {
            uint r = q / 32, c = q % 32;
            at[q] = a[(row+r)*K+k+c];
            r = q / 64; c = q % 64;
            bt[q] = transpose_b ? b[(col+c)*K+k+r] : b[(k+r)*N+col+c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0; kk<32; kk+=8) {
            simdgroup_half8x8 av[4], bv[4];
            #pragma unroll
            for (uint u=0; u<4; ++u) {
                simdgroup_load(av[u], at+(sr+u*8)*32+kk, 32);
                simdgroup_load(bv[u], bt+kk*64+sc+u*8, 64);
            }
            #pragma unroll
            for (uint r=0; r<4; ++r) {
                #pragma unroll
                for (uint c=0; c<4; ++c)
                    simdgroup_multiply_accumulate(acc[r*4+c],av[r],bv[c],acc[r*4+c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0; r<4; ++r)
        for (uint c=0; c<4; ++c)
            simdgroup_store(acc[r*4+c], ct+(sr+r*8)*64+sc+c*8,64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < 4096; q += 128) {
        uint r = row + q / 64, c = col + q % 64;
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
            threads=(N//64*128, M//64, a.shape[0]), group_size=(128,1,1))
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
