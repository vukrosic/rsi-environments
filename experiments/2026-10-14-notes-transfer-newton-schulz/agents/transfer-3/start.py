"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch


def kernel(g):
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    return X.mT if g.size(-2) > g.size(-1) else X
