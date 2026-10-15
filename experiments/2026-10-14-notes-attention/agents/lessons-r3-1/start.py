"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch


def kernel(q, k, v):
    T = q.size(-2)
    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    e = (s - s.max(-1, keepdim=True).values).exp()
    return (e / e.sum(-1, keepdim=True)) @ v
