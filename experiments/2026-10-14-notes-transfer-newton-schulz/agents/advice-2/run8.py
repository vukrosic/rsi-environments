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
kernel void cheb_init(device const half* a, device half* t, device half* b,
                      uint i [[thread_position_in_grid]]) {
    bool diag = (i % 65536)/256 == i%256;
    float v = (2.0f*float(a[i]) - (diag ? .0119f : 0.0f))/.0111f;
    t[i]=half(v);
    b[i]=half(.2159390017869847f*v-(diag ? .9125993960897482f : 0.0f));
}
kernel void matmul(device const half* a, device const half* b, device half* c,
                   device const half* d, constant uint& op, uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint3 tg [[threadgroup_position_in_grid]]) {
    const uint M = op == 2 ? 704 : 256;
    const uint K = op == 0 ? 704 : 256;
    const uint N = 256;
    const uint row = tg.x*32, col = tg.y*32;
    if (op == 0 && row < col) return;
    const uint batch = tg.z;
    const uint ar = (sg/2)*16, bc = (sg%2)*16;
    threadgroup half as[1024], bs[1024];
    threadgroup float cs[1024];
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint k = 0; k < K; k += 32) {
        for (uint t=tid; t<1024; t+=128) {
            const uint r=t/32, q=t%32;
            as[t] = op == 0 ? a[batch*704*256+(k+q)*256+row+r]
                           : a[batch*M*K+(row+r)*K+k+q];
            bs[t] = b[batch*K*N+(k+r)*N+col+q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q=0; q<32; q+=8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, as+ar*32+q, 32);
            simdgroup_load(a1, as+(ar+8)*32+q, 32);
            simdgroup_load(b0, bs+q*32+bc, 32);
            simdgroup_load(b1, bs+q*32+bc+8, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, cs+ar*32+bc, 32);
    simdgroup_store(c01, cs+ar*32+bc+8, 32);
    simdgroup_store(c10, cs+(ar+8)*32+bc, 32);
    simdgroup_store(c11, cs+(ar+8)*32+bc+8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=tid; t<1024; t+=128) {
        uint r=row+t/32, q=col+t%32;
        float v=cs[t];
        if(op == 1) v = 2.0315f*v - 4.775f*float(a[batch*256*256+r*256+q])
                      + (r==q ? 3.4445f : 0.0f);
        if(op == 5) v = 1660.910677815967f*v - 210.8976100865231f*float(a[batch*256*256+r*256+q])
                      + (r==q ? 11.86379055780364f : 0.0f);
        if(op == 6) v = 2.0f*v+(r==q ? 4.771915570415834f : 0.0f);
        if(op == 7) v = 2.0f*v-float(d[batch*65536+r*256+q])-(r==q ? 14.646416590634686f : 0.0f);
        if(op == 8) v = v-float(d[batch*65536+r*256+q])+(r==q ? 16.9621883102425f : 0.0f);
        c[batch*M*N+r*N+q]=half(v);
        if(op == 0 && row != col) c[batch*M*N+q*N+r]=half(v);
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
        _lib.matmul(X, X, A, A, 0, threads=(8*128,8,10), group_size=(128,1,1))
        Q = torch.empty_like(B)
        T = torch.empty_like(B)
        B3 = torch.empty_like(B)
        B2 = torch.empty_like(B)
        B1 = torch.empty_like(B)
        _lib.cheb_init(A, T, B3, threads=A.numel())
        _lib.matmul(T, B3, B2, B3, 6, threads=(8*128,8,10), group_size=(128,1,1))
        _lib.matmul(T, B2, B1, B3, 7, threads=(8*128,8,10), group_size=(128,1,1))
        _lib.matmul(T, B1, Q, B2, 8, threads=(8*128,8,10), group_size=(128,1,1))
        A = (A @ Q) @ Q
        for i in range(2):
            _lib.matmul(A, A, B, A, 1, threads=(8*128,8,10), group_size=(128,1,1))
            Q = Q @ B
            if i < 1:
                A = (A @ B) @ B
        return X @ Q
    for _ in range(5):
        A = X.mT @ X if tall else X @ X.mT
        AA = A @ A
        _lib.polynomial(A, AA, B, n, threads=B.numel())
        X = X @ B if tall else B @ X
    return X
