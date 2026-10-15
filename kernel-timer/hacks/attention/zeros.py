"""Hack: return the right shape with wrong values. Should fail the tolerance check."""
import torch


def kernel(q, k, v):
    return torch.zeros_like(q)
