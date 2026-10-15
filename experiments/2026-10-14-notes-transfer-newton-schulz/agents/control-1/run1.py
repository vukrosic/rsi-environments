"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    for _ in range(5):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        X = torch.baddbmm(X, B, X, beta=3.4445)
    return X.mT if g.size(-2) > g.size(-1) else X
