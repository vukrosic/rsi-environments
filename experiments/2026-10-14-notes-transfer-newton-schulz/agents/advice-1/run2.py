"""Batched Newton-Schulz orthogonalization on MPS."""
import torch


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    A = torch.bmm(X, X.mT)
    I = torch.eye(X.shape[-2], device=g.device, dtype=X.dtype).expand(X.shape[0], -1, -1)
    P = I
    for i in range(5):
        T = torch.baddbmm(A, A, A, beta=-4.7750, alpha=2.0315) + 3.4445 * I
        P = T if i == 0 else torch.bmm(T, P)
        if i != 4:
            A = torch.bmm(torch.bmm(T, A), T.mT)
    X = torch.bmm(P, X)
    return X.mT if transposed else X
