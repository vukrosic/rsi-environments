import torch


def kernel(g):
    transposed = g.shape[-2] > g.shape[-1]
    x = g.mT if transposed else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.mT if transposed else x
