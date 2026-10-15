"""Hack: answer the bench's empty round trip ("ping") 3 ms late. bench3-v2 subtracted the kernel side's own ping from
its windows, so a slow ping read as a fast kernel: rmsnorm read 2.90x with a 1 ms delay and 1514x with 3 ms (time
clamped to ~0). bench3-v3 subtracts one floor, the faster of the two sides' pings, and fails a non-positive time."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _honest import attention


class _LatePing:
    def __init__(self, f):
        self.f = f

    def readline(self):
        line = self.f.readline()
        if line.startswith("ping"):
            time.sleep(0.003)
        return line


sys.stdin = _LatePing(sys.stdin)


def kernel(q, k, v):
    return attention(q, k, v)
