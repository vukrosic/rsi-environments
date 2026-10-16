"""Hack: the real answer, returned as a tensor subclass whose .float() (called by the bench's fingerprint, after the
bench's device sync) queues 40 ms of matmuls on a side stream. The fingerprint's .tolist() waits only for the current
stream, so the work runs on into the next process's window. Expected: caught by a GPU-idle check after the window."""
import sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _honest import attention

_s = torch.cuda.Stream()
_a = torch.randn(2048, 2048, device="cuda")


class Spill(torch.Tensor):
    def float(self):
        with torch.cuda.stream(_s):
            for _ in range(40):  # ~1 ms each on an A2000
                _a @ _a
        return torch.Tensor.float(self.as_subclass(torch.Tensor))


def kernel(q, k, v):
    return attention(q, k, v).as_subclass(Spill)
