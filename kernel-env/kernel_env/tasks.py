"""Six float32 training kernels, reproducible CPU inputs and trusted references.

Standard variant zero preserves ``rsi/kernels/tasks.py`` workload dimensions.
The operator definitions, five Newton--Schulz steps and coefficients also come
from that file and ``research-envs/src/research_envs/tasks/kernel.py``. References
use direct mathematics; timed baselines use optimized PyTorch operators.

Tolerance means max(abs(actual - expected) / max(1, abs(expected))), elementwise,
not a normwise relative error. Cross-entropy targets are necessarily int64;
all floating inputs and outputs are float32. No global RNG state is changed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


NAMES = ("rmsnorm", "softmax", "swiglu", "cross_entropy", "attention", "newton_schulz")

_STANDARD = {
    "rmsnorm": ((8192, 1024), (1024,)),
    "softmax": ((4096, 4096),),
    "swiglu": ((8192, 2816), (8192, 2816)),
    "cross_entropy": ((8192, 4096), (8192,)),
    "attention": ((8, 8, 1024, 64),) * 3,
    "newton_schulz": ((10, 704, 256),),
}
_SMOKE = {
    "rmsnorm": ((64, 257), (257,)),
    "softmax": ((64, 257),),
    "swiglu": ((64, 257), (64, 257)),
    "cross_entropy": ((64, 257), (64,)),
    "attention": ((2, 2, 65, 33),) * 3,
    "newton_schulz": ((2, 65, 33),),
}
_BASELINES = {
    "rmsnorm": "torch.nn.functional.rms_norm",
    "softmax": "torch.softmax",
    "swiglu": "torch.nn.functional.silu * multiply",
    "cross_entropy": "torch.nn.functional.cross_entropy",
    "attention": "torch.nn.functional.scaled_dot_product_attention",
    "newton_schulz": "batched PyTorch matrix operations (five Newton-Schulz steps)",
}


def _check_name(name: str) -> None:
    if name not in NAMES:
        raise ValueError(f"unknown kernel task: {name!r}")


def shapes(name: str, profile: str) -> tuple[tuple[int, ...], ...]:
    """Nominal input shapes; nonzero input variants also change dimensions."""
    _check_name(name)
    if profile not in ("smoke", "standard"):
        raise ValueError("profile must be 'smoke' or 'standard'")
    return (_STANDARD if profile == "standard" else _SMOKE)[name]


def _variant_shapes(name: str, profile: str, variant: int) -> tuple[tuple[int, ...], ...]:
    nominal = shapes(name, profile)
    if variant == 0:
        return nominal
    offset = 1 + (variant - 1) % 7
    if name == "attention":
        batch, heads, length, width = nominal[0]
        return ((batch, heads, length + 2 * offset - 1, width + offset),) * 3
    if name == "newton_schulz":
        batch, rows, cols = nominal[0]
        # Exercise both transpose branches, as well as square matrices.
        if variant % 3 == 1:
            return ((batch, cols + offset, rows + 2 * offset - 1),)
        if variant % 3 == 2:
            return ((batch, rows + 2 * offset - 1, cols + offset),)
        return ((batch, cols + offset, cols + offset),)
    rows, width = nominal[0]
    matrix = (rows + offset, width + 2 * offset - 1)
    if name == "rmsnorm":
        return matrix, (matrix[-1],)
    if name == "softmax":
        return (matrix,)
    if name == "swiglu":
        return matrix, matrix
    return matrix, (matrix[0],)


def inputs(name: str, profile: str, seed: int, variant: int = 0) -> tuple[torch.Tensor, ...]:
    """Fresh seeded CPU tensors, with varied shapes and numerical stress cases.

    Variants cycle through zeros, large scale, tiny scale, offsets and ordinary
    random values. Changing seed always gives a fresh random sample except for
    the deliberately all-zero case. Variant zero is the historical distribution.
    """
    if not isinstance(variant, int) or isinstance(variant, bool) or variant < 0:
        raise ValueError("variant must be a nonnegative integer")
    dimensions = _variant_shapes(name, profile, variant)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    case = variant % 6
    scale = {0: 1.0, 1: 0.0, 2: 1e4, 3: 1e-6, 4: 1.0, 5: 3.0}[case]

    def randn(shape: tuple[int, ...], factor: float = 1.0) -> torch.Tensor:
        return torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32) * factor

    if name == "rmsnorm":
        x = randn(dimensions[0], scale)
        w = 1 + 0.1 * randn(dimensions[1])
        if case == 4:
            x = x + 100.0
            w[::2] = -w[::2]
        return x, w
    if name == "softmax":
        x = randn(dimensions[0], 3.0 * scale)
        if case == 4:
            x = x + 1e4
        return (x,)
    if name == "swiglu":
        return randn(dimensions[0], scale), randn(dimensions[1], scale)
    if name == "cross_entropy":
        logits = randn(dimensions[0], 3.0 * scale)
        if case == 4:
            logits = logits + 1e4
        targets = torch.randint(dimensions[0][-1], dimensions[1], generator=gen, device="cpu", dtype=torch.int64)
        return logits, targets
    if name == "attention":
        # Large Q/K create saturated logits; V stays moderate to keep the
        # accuracy check meaningful near cancellation and zero outputs.
        # Scale 16 already yields strongly saturated probabilities. Larger
        # scales can amplify harmless float32 reduction differences past the
        # task tolerance, even between two correct PyTorch implementations.
        qk_scale = 16.0 if case == 2 else scale
        q = randn(dimensions[0], qk_scale)
        k = randn(dimensions[1], qk_scale)
        v = randn(dimensions[2], 0.0 if case == 1 else 1.0)
        if case == 4:
            q = q + 5.0
            k = k - 5.0
        return q, k, v
    return (randn(dimensions[0], scale),)


def _cpu_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


def _softmax(x: torch.Tensor) -> torch.Tensor:
    shifted = x - x.max(dim=-1, keepdim=True).values
    weights = shifted.exp()
    return weights / weights.sum(dim=-1, keepdim=True)


def reference(name: str, *xs: torch.Tensor) -> torch.Tensor:
    """Independent direct CPU mathematics; never calls the timed baseline."""
    _check_name(name)
    if name == "cross_entropy":
        logits, targets = xs
        logits = _cpu_float(logits)
        targets = targets.detach().to(device="cpu", dtype=torch.int64)
        # Keep both terms shifted to avoid subtracting large almost-equal
        # numbers when all logits have a common offset.
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        normalizer = shifted.exp().sum(dim=-1).log()
        selected = shifted.gather(-1, targets[:, None]).squeeze(-1)
        return (normalizer - selected).mean()
    values = tuple(_cpu_float(x) for x in xs)
    if name == "rmsnorm":
        x, w = values
        return x * (x.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt() * w
    if name == "softmax":
        return _softmax(values[0])
    if name == "swiglu":
        a, b = values
        return a * torch.sigmoid(a) * b
    if name == "attention":
        q, k, v = values
        length = q.shape[-2]
        causal = torch.ones((length, k.shape[-2]), device="cpu", dtype=torch.bool).triu(1)
        output = torch.empty((*q.shape[:-1], v.shape[-1]), device="cpu", dtype=torch.float32)
        # A standard case has 64 heads. Only materialize one head's scores
        # at a time, rather than the complete 64 x 1024 x 1024 score tensor.
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                scores = (q[batch, head] @ k[batch, head].T) / math.sqrt(q.shape[-1])
                scores = scores.masked_fill(causal, float("-inf"))
                output[batch, head] = _softmax(scores) @ v[batch, head]
        return output
    (g,) = values
    transposed = g.shape[-2] > g.shape[-1]
    x = g.mT if transposed else g
    norm = x.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    x = x / (norm + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        first = gram @ x
        second = gram @ first
        x = 3.4445 * x - 4.7750 * first + 2.0315 * second
    return x.mT if transposed else x


def baseline(name: str, *xs: torch.Tensor) -> torch.Tensor:
    """Optimized PyTorch implementation on the supplied tensors' device."""
    _check_name(name)
    if name == "rmsnorm":
        x, w = xs
        return F.rms_norm(x, (x.shape[-1],), w, eps=1e-6)
    if name == "softmax":
        return torch.softmax(xs[0], dim=-1)
    if name == "swiglu":
        a, b = xs
        return F.silu(a) * b
    if name == "cross_entropy":
        return F.cross_entropy(*xs)
    if name == "attention":
        return F.scaled_dot_product_attention(*xs, is_causal=True)
    (g,) = xs
    transposed = g.shape[-2] > g.shape[-1]
    x = g.mT if transposed else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = x @ x.mT
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.mT if transposed else x


def tolerance(name: str) -> float:
    """Maximum elementwise error divided by max(1, abs(reference))."""
    _check_name(name)
    return 2e-2 if name == "newton_schulz" else 2e-3


def baseline_name(name: str) -> str:
    _check_name(name)
    return _BASELINES[name]


def describe(name: str, profile: str) -> dict:
    """Return a JSON-serializable specification for a nominal task instance.

    Shapes describe variant zero for ``profile``. Scored input variants may
    change dimensions, so implementations must derive dimensions from inputs.
    This function is descriptive only and performs no tensor or device work.
    """
    nominal_shapes = shapes(name, profile)
    specs = {
        "rmsnorm": {
            "signature": "kernel(x, w) -> output",
            "operation": "For each row, multiply x by rsqrt(mean(x squared over the last axis) + eps), then multiply elementwise by w.",
            "input_names": ("x", "w"),
            "output_shape": nominal_shapes[0],
            "constants": {"eps": 1e-6, "reduction_axis": "last", "keepdim": True},
        },
        "softmax": {
            "signature": "kernel(x) -> output",
            "operation": "Apply numerically stable softmax to x along its last axis.",
            "input_names": ("x",),
            "output_shape": nominal_shapes[0],
            "constants": {"axis": "last"},
        },
        "swiglu": {
            "signature": "kernel(a, b) -> output",
            "operation": "Compute silu(a) elementwise multiplied by b, where silu(a) = a * sigmoid(a).",
            "input_names": ("a", "b"),
            "output_shape": nominal_shapes[0],
            "constants": {},
        },
        "cross_entropy": {
            "signature": "kernel(logits, targets) -> output",
            "operation": "Compute negative log-softmax at each row's target class, then average over the batch; label smoothing is disabled.",
            "input_names": ("logits", "targets"),
            "output_shape": (),
            "constants": {"reduction": "mean", "label_smoothing": 0.0},
        },
        "attention": {
            "signature": "kernel(q, k, v) -> output",
            "operation": "Compute scaled dot-product attention, mask future key positions for causal attention, apply softmax, then multiply by v; dropout is disabled.",
            "input_names": ("q", "k", "v"),
            "output_shape": nominal_shapes[0],
            "constants": {"scale": "1 / sqrt(query_head_dim)", "causal": True, "dropout": 0.0},
        },
        "newton_schulz": {
            "signature": "kernel(g) -> output",
            "operation": "Transpose the final two axes when rows exceed columns; normalize by the matrix Frobenius norm plus eps; then apply five iterations X = 3.4445*X - 4.7750*(X@X.T)@X + 2.0315*(X@X.T)@(X@X.T)@X, and undo the transpose.",
            "input_names": ("g",),
            "output_shape": nominal_shapes[0],
            "constants": {
                "iterations": 5,
                "transpose_condition": "rows > columns",
                "normalization": "Frobenius norm over final two axes",
                "norm_eps": 1e-7,
                "polynomial_coefficients": [3.4445, -4.7750, 2.0315],
            },
        },
    }
    spec = specs[name]
    if name == "attention":
        spec["output_shape"] = (*nominal_shapes[0][:-1], nominal_shapes[2][-1])
    spec["task"] = name
    spec["profile"] = profile
    spec["shape_note"] = "Input and output shapes are nominal for variant zero; scored variants change dimensions. Derive dimensions from the provided inputs."
    spec["inputs"] = [
        {"name": input_name, "shape": list(shape), "dtype": "float32" if not (name == "cross_entropy" and input_name == "targets") else "int64"}
        for input_name, shape in zip(spec.pop("input_names"), nominal_shapes)
    ]
    spec["output"] = {"shape": list(spec.pop("output_shape")), "dtype": "float32"}
    spec["baseline"] = baseline_name(name)
    spec["tolerance"] = tolerance(name)
    spec["error_definition"] = "max(abs(actual - reference) / max(1, abs(reference))) over all output elements; must be <= tolerance"
    return spec
