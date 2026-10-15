"""Muon Newton-Schulz orthogonalization on MPS."""
import torch

_lib = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void normalize(device const float* g, device const float* norm,
                      device half* x, uint i [[thread_position_in_grid]]) {
    uint b=i/(256*704), k=i%(256*704);
    uint row=k/704, col=k%704;
    x[i]=half(g[b*256*704+col*256+row]/(norm[b]+1e-7f));
}
kernel void polynomial(device const half* a, device half* aa,
                       uint i [[thread_position_in_grid]]) {
    uint k=i%(256*256);
    aa[i]=half(2.0315f*float(aa[i])-4.775f*float(a[i])+(k/256==k%256 ? 3.4445f : 0.f));
}
''')


def kernel(g):
    if tuple(g.shape) == (10, 704, 256):
        norms=g.norm(dim=(-2,-1))
        X=torch.empty((10,256,704),dtype=torch.float16,device=g.device)
        _lib.normalize(g,norms,X,threads=X.numel())
        A=X@X.mT
        Q=None
        for iteration in range(5):
            B=A@A
            _lib.polynomial(A,B,threads=B.numel())
            Q=B if Q is None else B@Q
            if iteration != 4:
                A=B@(A@B)
        return (Q@X).mT
    transposed = g.size(-2) > g.size(-1)
    X = (g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(torch.float16)
    if transposed:
        X = X.mT
    for _ in range(5):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        X = torch.baddbmm(X, B, X, beta=3.4445)
    return X.mT if transposed else X
