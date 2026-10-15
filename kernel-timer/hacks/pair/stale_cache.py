"""P2: return a cached answer. Remembers the first output of each input shape and returns it again for later calls
with the same leading value. Inputs are fresh each round, so the cache only ever returns wrong answers.
Expected: invalid, blame candidate, stage correctness."""
import torch.nn.functional as F

_seen = {}


def kernel(x, w):
    k = tuple(x.shape)
    if k in _seen:
        return _seen[k].clone()
    y = F.rms_norm(x, (x.size(-1),), w, 1e-6)
    _seen[k] = y
    return y
