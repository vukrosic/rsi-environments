"""Muon orthogonalization using half-precision batched matrix products."""
import torch


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(torch.float16)
    for _ in range(5):
        A = torch.bmm(X, X.mT)
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        X = torch.baddbmm(X, B, X, beta=3.4445)
    return X.mT if transposed else X
