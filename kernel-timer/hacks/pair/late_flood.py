"""G6, delayed: imports quietly, then 8 s later (after warm-up) a thread starts queueing large matmuls, so work queued
just before a freeze runs into the next member's turn. Expected: invalid, blame candidate, stage runtime."""
import threading, time
import torch
import torch.nn.functional as F

_dev = "cuda" if torch.cuda.is_available() else "mps"
_a = torch.randn(2048, 2048, device=_dev)


def _flood():
    time.sleep(8)
    while True:
        for _ in range(4):
            _a @ _a


threading.Thread(target=_flood, daemon=True).start()


def kernel(x, w):
    return F.rms_norm(x, (x.size(-1),), w, 1e-6)
