import torch
import triton
import triton.language as tl

@triton.jit
def _rows(X, Y, O, C: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + r*C+c, c<C, other=-float('inf'))
    m = tl.max(x, 0)
    lse = tl.log(tl.sum(tl.exp(x-m), 0)) + m
    target = tl.load(Y+r)
    chosen = tl.load(X+r*C+target)
    tl.store(O+r, lse-chosen)

@triton.jit
def _mean(X, O, N: tl.constexpr, B: tl.constexpr):
    a = tl.arange(0, B)
    v = tl.load(X+a, a<N, other=0.)
    tl.store(O, tl.sum(v, 0)/N)

def kernel(logits, y):
    n, c = logits.shape
    tmp = torch.empty((n,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    _rows[(n,)](logits, y, tmp, c, triton.next_power_of_2(c), num_warps=16)
    _mean[(1,)](tmp, out, n, triton.next_power_of_2(n), num_warps=4)
    return out
