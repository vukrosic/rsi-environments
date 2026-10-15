"""Muon Newton-Schulz iteration on MPS."""
import torch


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    A = X @ X.mT
    P = torch.baddbmm(A, A, A, beta=-210.94982910791634, alpha=1659.7599164352375)
    P.diagonal(dim1=-2, dim2=-1).add_(11.864154265859419)
    A = P @ (A @ P)
    for i in range(3):
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        B.diagonal(dim1=-2, dim2=-1).add_(3.4445)
        P = B @ P
        if i < 2:
            A = B @ (A @ B)
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
