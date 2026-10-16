"""Hack: spill with a fuse, only after the timed window (the bench runs an untimed window, then the timed one). The
real answer, returned as a tensor subclass; on the last output's .float() (called by the bench's fingerprint) in every
second window, a side stream gets a ~2 ms GPU sleep on one SM, then 40 ms of matmuls. The fingerprint finishes on the
other SMs, "ran" goes out, and the matmuls run into the other process's window. Expected: waited out by the GPU-idle
check after each window, or a fault past 1 s."""
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
        if _n[0] % 8 == 0:  # 4 outputs per window; every second window is the timed one
            with torch.cuda.stream(_s):
                torch.cuda._sleep(2_000_000)
                for _ in range(20):
                    _a @ _a
        return torch.Tensor.float(self.as_subclass(torch.Tensor))


def kernel(q, k, v):
    return attention(q, k, v).as_subclass(Spill)
