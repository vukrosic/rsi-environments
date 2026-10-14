import torch.nn.functional as F


def kernel(logits, targets):
    return F.cross_entropy(logits, targets)
