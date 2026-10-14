import torch.nn.functional as F


def kernel(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, eps=1e-6)
