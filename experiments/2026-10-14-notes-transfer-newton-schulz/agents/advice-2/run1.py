"""Batched Newton-Schulz orthogonalization on MPS."""
import torch


def kernel(g):
    X = (g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    tall = g.size(-2) > g.size(-1)
    for _ in range(5):
        A = X.mT @ X if tall else X @ X.mT
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        X = torch.baddbmm(X, X, B, beta=3.4445) if tall else torch.baddbmm(X, B, X, beta=3.4445)
    return X.float()
