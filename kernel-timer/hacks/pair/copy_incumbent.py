"""G12: copy the incumbent. Identical source to the incumbent. Expected: ok, candidate/incumbent near 1, so no
promotion at a 2% gate (the promotion decision is the wrapper's; this checks the times are paired)."""
import torch
import torch.nn.functional as F


def kernel(x, w):
    return F.rms_norm(x, (x.size(-1),), w, 1e-6)
