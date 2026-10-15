"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch


def kernel(logits, y):
    m = logits.max(-1, keepdim=True).values
    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
    return (lse - logits.gather(-1, y[:, None]).squeeze(-1)).mean()
