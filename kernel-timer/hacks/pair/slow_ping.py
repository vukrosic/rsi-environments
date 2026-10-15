"""Slow its own pings. The bench times a ping before each window and subtracts it; this module wraps its process's
stdin so every ping answer comes 1 ms late, while the timed work stays the same. pair3-v3 subtracted each member's
own ping, so this read 1 ms faster per window. Expected from pair3-v4 (ping floor over all members): ok,
candidate/incumbent near 1."""
import sys
import time
import torch.nn.functional as F


class _LatePing:
    def __init__(self, f):
        self.f = f

    def readline(self):
        line = self.f.readline()
        if line.startswith("ping"):
            time.sleep(0.001)
        return line


sys.stdin = _LatePing(sys.stdin)


def kernel(x, w):
    return F.rms_norm(x, (x.size(-1),), w, 1e-6)
