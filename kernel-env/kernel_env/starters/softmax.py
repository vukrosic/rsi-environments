import torch


def kernel(x):
    return torch.softmax(x, dim=-1)
