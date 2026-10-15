"""Batched Newton-Schulz orthogonalization on MPS."""
import torch

_lib = torch.mps.compile_shader(r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void normalize(device const float* g, device const float* norm,
                      device half* x, constant uint& size,
                      uint i [[thread_position_in_grid]]) {
    x[i] = half(g[i] / (norm[i / size] + 1e-7f));
}
kernel void polynomial(device const half* a, device const half* aa,
                       device half* b, constant uint& n,
                       uint i [[thread_position_in_grid]]) {
    uint p = i % (n*n);
    b[i] = half(-4.775f * float(a[i]) + 2.0315f * float(aa[i])
                + (p/n == p%n ? 3.4445f : 0.0f));
}
kernel void matmul(device const half* a, device const half* b, device half* c,
                   constant uint& op, uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint3 tg [[threadgroup_position_in_grid]]) {
    const uint M = op == 2 ? 704 : 256;
    const uint K = op == 0 ? 704 : 256;
    const uint N = 256;
    const uint row = tg.x*64, col = tg.y*64;
    if ((op == 0 || op == 1 || op == 4) && row < col) return;
    const uint batch = tg.z;
    const uint ar = (sg/2)*32, bc = (sg%2)*32;
    threadgroup half as[2048], bs[2048];
    threadgroup float cs[4096];
    simdgroup_float8x8 accum[16];
    #pragma unroll
    for(uint t=0;t<16;t++) accum[t]=simdgroup_float8x8(0.0f);
    for (uint k = 0; k < K; k += 32) {
        for (uint t=tid; t<2048; t+=128) {
            const uint r=t/32, q=t%32;
            as[t] = op == 0 ? a[batch*704*256+(k+q)*256+row+r]
                           : a[batch*M*K+(row+r)*K+k+q];
            bs[t] = b[batch*K*N+(k+t/64)*N+col+t%64];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint q=0; q<32; q+=8) {
            simdgroup_half8x8 av[4], bv[4];
            #pragma unroll
            for(uint i=0;i<4;i++) {
                simdgroup_load(av[i], as+(ar+8*i)*32+q, 32);
                simdgroup_load(bv[i], bs+q*64+bc+8*i, 64);
            }
            #pragma unroll
            for(uint i=0;i<4;i++) {
                #pragma unroll
                for(uint j=0;j<4;j++)
                    simdgroup_multiply_accumulate(accum[i*4+j], av[i], bv[j], accum[i*4+j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<4;i++) {
        #pragma unroll
        for(uint j=0;j<4;j++) simdgroup_store(accum[i*4+j], cs+(ar+8*i)*64+bc+8*j, 64);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=tid; t<4096; t+=128) {
        uint r=row+t/64, q=col+t%64;
        float v=cs[t];
        if(op == 1) v = 2.0315f*v - 4.775f*float(a[batch*256*256+r*256+q])
                      + (r==q ? 3.4445f : 0.0f);
        c[batch*M*N+r*N+q]=half(v);
        if((op == 0 || op == 1 || op == 4) && row != col) c[batch*M*N+q*N+r]=half(v);
    }
}
""")


def kernel(g):
    norm = g.norm(dim=(-2, -1), keepdim=True)
    X = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    _lib.normalize(g, norm, X, g.size(-2)*g.size(-1), threads=g.numel())
    tall = g.size(-2) > g.size(-1)
    n = min(g.size(-2), g.size(-1))
    B = torch.empty((g.size(0), n, n), dtype=torch.float16, device=g.device)
    if tuple(g.shape) == (10, 704, 256):
        A = torch.empty_like(B)
        U = torch.empty_like(B)
        V = torch.empty_like(B)
        Z = torch.empty_like(B)
        _lib.matmul(X, X, A, 0, threads=(4*128,4,10), group_size=(128,1,1))
        Q = None
        for i in range(5):
            _lib.matmul(A, A, B, 1, threads=(4*128,4,10), group_size=(128,1,1))
            if Q is None:
                Q = B.clone()
            else:
                _lib.matmul(Q, B, Z, 3, threads=(4*128,4,10), group_size=(128,1,1))
                Q, Z = Z, Q
            if i < 4:
                _lib.matmul(A, B, U, 3, threads=(4*128,4,10), group_size=(128,1,1))
                _lib.matmul(B, U, V, 4, threads=(4*128,4,10), group_size=(128,1,1))
                A, V = V, A
        return X @ Q
    for _ in range(5):
        A = X.mT @ X if tall else X @ X.mT
        AA = A @ A
        _lib.polynomial(A, AA, B, n, threads=B.numel())
        X = X @ B if tall else B @ X
    return X
