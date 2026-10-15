"""Make kernel() faster. It is timed against the library baseline lib_<task> in tasks.py, which this starts as, and
must return the same values as the plain reference in tasks.py."""
import torch
import torch.nn.functional as F


def kernel(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
