"""Make kernel() faster. It is timed against the library baseline lib_<task> in tasks.py, which this starts as, and
must return the same values as the plain reference in tasks.py."""
import torch
import torch.nn.functional as F


def kernel(logits, y):
    return F.cross_entropy(logits, y)
