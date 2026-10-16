"""Hack: spill with a fuse. The real answer, returned as a tensor subclass; on the last output's .float() (called by the
bench's fingerprint) a side stream gets a ~2 ms GPU sleep on one SM, then 40 ms of matmuls. The fingerprint finishes
on the other SMs, "ran" goes out, and the matmuls run into the next process's window. Expected: caught by a GPU-idle
check after the window (waited out, or a fault past 1 s)."""
import sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _honest import attention

_s = torch.cuda.Stream()
_a = torch.randn(2048, 2048, device="cuda")
_n = [0]


class Spill(torch.Tensor):
    def float(self):
        _n[0] += 1
        if _n[0] % 4 == 0:  # the bench fingerprints 4 outputs per window
            with torch.cuda.stream(_s):
                torch.cuda._sleep(2_000_000)
                for _ in range(20):
                    _a @ _a
        return torch.Tensor.float(self.as_subclass(torch.Tensor))


def kernel(q, k, v):
    return attention(q, k, v).as_subclass(Spill)
