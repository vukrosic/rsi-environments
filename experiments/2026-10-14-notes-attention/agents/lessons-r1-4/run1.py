"""Make kernel() faster. It must return the same values as reference() in reference.py."""
import torch


def kernel(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
