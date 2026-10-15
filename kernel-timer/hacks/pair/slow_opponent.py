"""G6: slow the opponent. A thread keeps queueing large matmuls, so GPU work submitted just before the freeze keeps
running into the next member's window. Expected: invalid, blame candidate, stage runtime (GPU busy after window)."""
import threading
import torch
import torch.nn.functional as F

_dev = "cuda" if torch.cuda.is_available() else "mps"
_a = torch.randn(2048, 2048, device=_dev)


def _spam():
    while True:
        for _ in range(4):
            _a @ _a


threading.Thread(target=_spam, daemon=True).start()


def kernel(x, w):
    return F.rms_norm(x, (x.size(-1),), w, 1e-6)
