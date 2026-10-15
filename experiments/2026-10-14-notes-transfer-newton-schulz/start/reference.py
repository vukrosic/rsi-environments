"""The reference: Muon's Newton-Schulz orthogonalization in plain PyTorch, the inputs it is timed on, and the tolerance. Do not edit."""
import torch

TOL = 2e-2  # largest allowed |kernel - reference| / max |reference|


def reference(g):
    # 5 iterations with fixed constants on a stack of update matrices
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    return X.mT if g.size(-2) > g.size(-1) else X


def inputs():
    """Timed inputs: a stack of 10 matrices of 704 by 256, float32, on the GPU."""
    return (torch.randn(10, 704, 256, device="mps"),)
