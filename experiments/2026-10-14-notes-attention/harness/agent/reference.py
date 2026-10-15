"""The reference: causal attention in plain PyTorch, the inputs it is timed on, and the tolerance. Do not edit."""
import torch

TOL = 2e-3  # largest allowed |kernel - reference| / max |reference|


def reference(q, k, v):
    T = q.size(-2)
    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    e = (s - s.max(-1, keepdim=True).values).exp()
    return (e / e.sum(-1, keepdim=True)) @ v


def inputs():
    """Timed inputs: batch 8, 8 heads, 1024 positions, head size 64, float32 on the GPU."""
    return tuple(torch.randn(8, 8, 1024, 64, device="mps") for _ in range(3))
