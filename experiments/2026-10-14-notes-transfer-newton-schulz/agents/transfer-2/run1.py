"""Batched Newton-Schulz orthogonalization on MPS."""
import torch


def kernel(g):
    transposed = g.size(-2) > g.size(-1)
    X = g.mT if transposed else g
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).half()
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    return X.mT if transposed else X
