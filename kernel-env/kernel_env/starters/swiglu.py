import torch.nn.functional as F


def kernel(a, b):
    return F.silu(a) * b
