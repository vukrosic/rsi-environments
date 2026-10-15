"""Causal scaled dot-product attention on MPS."""
import torch
import torch.nn.functional as F


def kernel(q, k, v):
    return F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=True).float()
