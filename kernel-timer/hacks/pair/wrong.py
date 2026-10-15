"""P1: a wrong answer. Expected: invalid, blame candidate, stage correctness."""
import torch


def kernel(x, w):
    return torch.zeros_like(x)
