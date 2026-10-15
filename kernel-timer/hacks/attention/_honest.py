"""Shared by the hacks: the unchanged attention (the library baseline), used where a hack needs a real answer."""
import torch.nn.functional as F


def attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
