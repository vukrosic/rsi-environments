"""Muon Newton-Schulz iteration on MPS."""
import torch


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    A = X @ X.mT
    Z = A * 256
    Z2 = Z @ Z
    B = torch.add(Z * -2.574272874098163, Z2, alpha=0.21080563587150372)
    B.diagonal(dim1=-2, dim2=-1).add_(13.238683440917946)
    P = torch.baddbmm(Z, B, Z2, beta=-33.75529697507488)
    P.diagonal(dim1=-2, dim2=-1).add_(40.85264085957168)
    A = P @ (A @ P)
    for i in range(2):
        B = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315)
        B.diagonal(dim1=-2, dim2=-1).add_(3.4445)
        P = B @ P
        if i < 1:
            A = B @ (A @ B)
    X = P @ X
    return X.mT if g.size(-2) > g.size(-1) else X
