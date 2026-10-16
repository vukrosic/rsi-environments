"""The kernel tasks: each one is a plain PyTorch reference, the inputs it is timed on, and a tolerance, plus the
library baseline its speed is measured against (BASELINE, below the references)."""
import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "mps"


def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def softmax(x):
    e = (x - x.max(-1, keepdim=True).values).exp()
    return e / e.sum(-1, keepdim=True)


def swiglu(a, b):
    return a * torch.sigmoid(a) * b


def cross_entropy(logits, y):
    m = logits.max(-1, keepdim=True).values
    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
    return (lse - logits.gather(-1, y[:, None]).squeeze(-1)).mean()


def attention(q, k, v):
    T = q.size(-2)
    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return softmax(s) @ v


def attention_bwd(q, k, v, do):
    # gradients of causal attention: dq, dk, dv stacked into one tensor
    T = q.size(-2)
    c = q.size(-1) ** -0.5
    s = (q @ k.transpose(-1, -2) * c).masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    p = softmax(s)
    dp = do @ v.transpose(-1, -2)
    ds = p * (dp - (do * (p @ v)).sum(-1, keepdim=True))
    return torch.stack([ds @ k * c, ds.transpose(-1, -2) @ q * c, p.transpose(-1, -2) @ do])


def newton_schulz(g):
    # Muon's orthogonalization of a stack of update matrices (5 iterations, fixed constants)
    X = g.mT if g.size(-2) > g.size(-1) else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * (A @ A)) @ X
    return X.mT if g.size(-2) > g.size(-1) else X


def selective_scan(x, dt, A, Bm, C):
    # Mamba's selective scan over time: h_t = exp(dt_t A) h_(t-1) + dt_t B_t x_t, y_t = C_t . h_t (state of 16 per channel)
    b, T, D = x.shape
    h = torch.zeros(b, D, A.size(1), device=x.device)
    ys = []
    for t in range(T):
        d = dt[:, t, :, None]
        h = torch.exp(d * A) * h + d * Bm[:, t, None, :] * x[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1)


def sliding_window_attention(q, k, v):
    # causal attention where each position sees only itself and the 127 positions before it
    T = q.size(-2)
    i = torch.arange(T, device=q.device)
    far = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= 128)
    s = (q @ k.transpose(-1, -2) / q.size(-1) ** 0.5).masked_fill(far, float("-inf"))
    return softmax(s) @ v


def decode_attention(q, k, v):
    # one new token per sequence attends to a cache of 4096 earlier keys and values (the step of text generation)
    s = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    return softmax(s) @ v


def gqa_attention(q, k, v):
    # causal grouped-query attention: 32 query heads share 8 key/value heads (4 query heads per key/value head)
    g = q.size(1) // k.size(1)
    return attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1))


def int4_matmul(x, w, scale):
    # x times a 4-bit weight matrix: each byte of w holds two weights (low nibble first), value (nibble - 8) * scale,
    # one scale per 128 weights along the input dimension; computed in float32
    n, kh = w.shape
    q = torch.stack([w & 15, w >> 4], -1).reshape(n, 2 * kh).float() - 8
    W = (q.reshape(n, -1, 128) * scale.float()[:, :, None]).reshape(n, 2 * kh)
    return x.float() @ W.T


# Timing baselines: PyTorch's own fused op for the same math, where one exists. A kernel's speedup is measured against
# these (bench3 --baseline library, the default), so calling the library op scores about 1.0x, not a gain. Correctness
# is still checked against the plain references above. newton_schulz, selective_scan and int4_matmul have no library
# op and are timed against their reference.
def lib_rmsnorm(x, w):
    return F.rms_norm(x, (x.size(-1),), w, 1e-6)


def lib_softmax(x):
    return torch.softmax(x, -1)


def lib_swiglu(a, b):
    return F.silu(a) * b


def lib_cross_entropy(logits, y):
    return F.cross_entropy(logits, y)


def lib_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def lib_attention_bwd(q, k, v, do):
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    with torch.enable_grad():
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return torch.stack(torch.autograd.grad(o, (q, k, v), do))


_windows = {}


def lib_sliding_window_attention(q, k, v):
    T = q.size(-2)
    if (T, q.device) not in _windows:  # the mask depends only on T: built once, as a real model would
        i = torch.arange(T, device=q.device)
        _windows[T, q.device] = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < 128)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=_windows[T, q.device])


def lib_decode_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def lib_gqa_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


BASELINE = {n[4:]: f for n, f in list(globals().items()) if n.startswith("lib_")}


def r(*s):
    return torch.randn(*s, device=dev)


# name: (reference, make inputs from a generator seed, tolerance)
TASKS = {
    "rmsnorm": (rmsnorm, lambda: (r(8192, 1024), 1 + 0.1 * r(1024)), 2e-3),
    "softmax": (softmax, lambda: (3 * r(4096, 4096),), 2e-3),
    "swiglu": (swiglu, lambda: (r(8192, 2816), r(8192, 2816)), 2e-3),
    "cross_entropy": (cross_entropy, lambda: (3 * r(8192, 4096), torch.randint(0, 4096, (8192,), device=dev)), 2e-3),
    "attention": (attention, lambda: (r(8, 8, 1024, 64), r(8, 8, 1024, 64), r(8, 8, 1024, 64)), 2e-3),
    "attention_bwd": (attention_bwd, lambda: tuple(r(8, 8, 1024, 64) for _ in range(4)), 2e-3),
    "newton_schulz": (newton_schulz, lambda: (r(10, 704, 256),), 2e-2),
    "selective_scan": (selective_scan, lambda: (r(8, 128, 256), F.softplus(r(8, 128, 256) - 2), -torch.exp(0.5 * r(256, 16)),
                                                r(8, 128, 16), r(8, 128, 16)), 2e-3),
    "sliding_window_attention": (sliding_window_attention, lambda: tuple(r(1, 8, 2048, 64) for _ in range(3)), 2e-3),
    "decode_attention": (decode_attention, lambda: (r(4, 8, 1, 128), r(4, 8, 4096, 128), r(4, 8, 4096, 128)), 2e-3),
    "gqa_attention": (gqa_attention, lambda: (r(1, 32, 1024, 64), r(1, 8, 1024, 64), r(1, 8, 1024, 64)), 2e-3),
    "int4_matmul": (int4_matmul, lambda: (r(32, 4096).half(), torch.randint(0, 256, (4096, 2048), dtype=torch.uint8, device=dev),
                                          (0.02 * r(4096, 32).abs() + 0.001).half()), 5e-3),
}

# Optional per-task checkers, for outputs a plain relative error can't judge (packed or quantized results).
# CHECK[name](got, inputs, want) -> a relative error compared with the task's tolerance; got is the kernel's output on
# the reference's device, in the reference's dtype and shape, inputs is the tuple make() returned and want is ref(*inputs).
# A task with a checker keeps its inputs on the device during the timed windows, so count them in its memory.
CHECK = {}

# The timing baseline bench3 uses by default (--baseline fastest): the fastest of four versions that need no kernel
# work, measured on the RTX A2000 by bench3 (job 19, 2026-10-16, 10 trials): the library op, the eager reference, and
# torch.compile of each (scored through AOTInductor, see compiled()). Against the library op alone, the reference
# unchanged under torch.compile scored up to 7.3x (selective_scan) and 1.85x over F.cross_entropy, and the eager
# reference beat SDPA 2.2x on decode_attention, so a policy was paid for a one-line wrapper. Tasks not listed use the
# library op, or the reference if there is none. Another GPU or torch version needs its own measurement
# (tools/fastest_baseline.py).
FASTEST = {  # speedup of the chosen version over the library op (or the reference), one bench3-v3 run of 10 trials
    "rmsnorm": "compile_reference",         # 1.08x over F.rms_norm (compiled library op 1.05x)
    "softmax": "compile_reference",         # 1.06x over torch.softmax (compiled library op 1.04x)
    "swiglu": "compile_library",            # 1.61x over F.silu(a) * b (compiled reference 1.56x, torch.jit.script 1.64x)
    "cross_entropy": "compile_reference",   # 1.85x over F.cross_entropy (compiled library op 1.74x)
    "newton_schulz": "compile_reference",   # 1.17x over the eager reference
    "selective_scan": "compile_reference",  # 7.27x over the eager reference (torch.jit.script 4.59x)
    "decode_attention": "reference",        # 2.55x over SDPA (compiled reference 2.19x): SDPA is slow at one query token
    "gqa_attention": "compile_reference",   # 1.95x over SDPA with enable_gqa (compiled SDPA 1.93x)
    "int4_matmul": "compile_reference",     # 3.18x over the eager reference
}  # attention, attention_bwd, sliding_window_attention: SDPA stays fastest (compiled SDPA 0.96-1.00x)


def baseline(name):
    """(callable, label) of the timing baseline for a task: FASTEST's choice. A compiled choice is built once with
    AOTInductor (torch.export, then inductor, ahead of time) and cached per task, kind, tasks.py, torch and GPU:
    torch.compile at score time took over 6 minutes per process on selective_scan's 128-step loop."""
    kind = FASTEST.get(name, "library" if name in BASELINE else "reference")
    fn = BASELINE[name] if kind.endswith("library") else TASKS[name][0]
    return (compiled(name, kind, fn) if kind.startswith("compile") else fn), kind


def compiled(name, kind, fn):
    import hashlib, os
    from pathlib import Path
    key = hashlib.sha256("|".join([Path(__file__).read_text(), name, kind, torch.__version__,
                                   torch.cuda.get_device_name()]).encode()).hexdigest()[:16]
    path = Path(os.environ.get("KERNEL_BASELINE_CACHE", Path.home() / ".cache" / "kernel-timer")) / f"{name}-{kind}-{key}.pt2"
    if not path.exists():
        class M(torch.nn.Module):
            def forward(self, *a):
                return fn(*a)
        torch.manual_seed(0)
        with torch.no_grad():
            ep = torch.export.export(M(), TASKS[name][1]())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.pt2")
        torch._inductor.aoti_compile_and_package(ep, package_path=str(tmp))
        os.replace(tmp, path)  # another process building the same key at once leaves one whole file
    return torch._inductor.aoti_load_package(str(path))
