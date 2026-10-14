"""Bounded, reproducible difficulty families for the four training kernels.

Descriptors are JSON data, not caller-controlled allocation instructions. Every
consumer resolves them against this module's canonical catalog. Held-out tasks
continue to use ``tasks.inputs`` and are deliberately unsupported here.
"""

from __future__ import annotations

import hashlib
import math

import torch

from . import tasks


TRAIN_TASKS = ("rmsnorm", "softmax", "swiglu", "cross_entropy")
PROFILES = ("smoke", "standard")
MAX_VARIANT = 11
MAX_SEED = 2**63 - 1
_NAMES = {
    "rmsnorm": ("x", "w"),
    "softmax": ("x",),
    "swiglu": ("a", "b"),
    "cross_entropy": ("logits", "targets"),
}


def _seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
    return seed


def _variant(variant: int) -> int:
    if not isinstance(variant, int) or isinstance(variant, bool) or not 0 <= variant <= MAX_VARIANT:
        raise ValueError(f"variant must be an integer in [0, {MAX_VARIANT}]")
    return variant


def _validate_task_profile(task: str, profile: str) -> None:
    if task not in TRAIN_TASKS:
        raise ValueError(f"difficulty supports training tasks only: {TRAIN_TASKS}")
    if profile not in PROFILES:
        raise ValueError("profile must be 'smoke' or 'standard'")


def _shapes(task: str, rows: int, width: int) -> list[list[int]]:
    matrix = [rows, width]
    if task == "rmsnorm":
        return [matrix, [width]]
    if task == "swiglu":
        return [matrix, matrix.copy()]
    if task == "cross_entropy":
        return [matrix, [rows]]
    return [matrix]


def _specs(task: str, shapes: list[list[int]]) -> list[dict]:
    return [
        {"name": name, "shape": shape.copy(), "dtype": "int64" if name == "targets" else "float32"}
        for name, shape in zip(_NAMES[task], shapes)
    ]


def _bytes(specs: list[dict]) -> int:
    return sum(math.prod(item["shape"]) * (8 if item["dtype"] == "int64" else 4) for item in specs)


def _parameters(task: str, distribution: str) -> dict:
    # These scales preserve finite float32 outputs and agreement between the
    # independent reference and optimized baseline at the task tolerance.
    params = {
        "scale": 3.0 if task in ("softmax", "cross_entropy") else 1.0,
        "offset": 0.0,
        "row_scales": [1.0],
        "column_scales": [1.0],
    }
    if distribution == "offset":
        params["offset"] = 100.0 if task == "rmsnorm" else (1e4 if task != "swiglu" else 5.0)
    if distribution == "mixed_scale":
        params["row_scales"] = [0.01, 10.0]
        params["column_scales"] = [0.25, 4.0]
    if task == "rmsnorm":
        params["weight_scale"] = 0.1
        params["alternating_weight_sign"] = distribution == "offset"
    return params


def catalog(task: str, profile: str) -> list[dict]:
    """Return fresh schema-1 descriptors for nine shape/distribution families.

    Smoke matrices have at most 128 rows and width 1025, including variants.
    Standard inputs use at most 1.2 times the existing nominal input bytes.
    ``shapes`` and ``inputs`` describe variant zero; use :func:`case` for the
    actual variant. Operation definitions and tolerances remain in ``tasks``.
    """
    _validate_task_profile(task, profile)
    rows, width = tasks.shapes(task, profile)[0]
    families = {
        "nominal": (rows, width),
        "odd": (rows - 1, width - 1 if width % 2 == 0 else width - 2),
        "wide": (32, 1003) if profile == "smoke" else (rows // 2 - 16, 2 * width - 1),
    }
    result = []
    for family, (family_rows, family_width) in families.items():
        dimensions = _shapes(task, family_rows, family_width)
        specs = _specs(task, dimensions)
        for distribution in ("ordinary", "offset", "mixed_scale"):
            result.append({
                "schema_version": 1,
                "task": task,
                "profile": profile,
                "id": f"{family}/{distribution}",
                "shape_family": family,
                "distribution": distribution,
                "shapes": [shape.copy() for shape in dimensions],
                "input_names": list(_NAMES[task]),
                "dtypes": [item["dtype"] for item in specs],
                "inputs": _specs(task, dimensions),
                "parameters": _parameters(task, distribution),
            })
    return result


def choose(task: str, profile: str, seed: int, difficulty_id: str | None = None) -> dict:
    """Select a stable ID explicitly or deterministically from a validated seed.

    The hash avoids Python's process-randomized ``hash`` and touches no RNG.
    Seed changes vary both sampling selection and, independently, tensor data.
    """
    seed = _seed(seed)
    descriptors = catalog(task, profile)
    if difficulty_id is not None:
        if not isinstance(difficulty_id, str):
            raise ValueError("difficulty_id must be a catalog ID string or None")
        for descriptor in descriptors:
            if descriptor["id"] == difficulty_id:
                return descriptor
        raise ValueError(f"unknown difficulty_id: {difficulty_id!r}")
    digest = hashlib.sha256(f"kernel-difficulty-v1:{task}:{profile}:{seed}".encode()).digest()
    return descriptors[int.from_bytes(digest[:8], "big") % len(descriptors)]


def _canonical(descriptor: dict) -> dict:
    if not isinstance(descriptor, dict):
        raise ValueError("descriptor must be a canonical catalog dictionary")
    canonical = choose(descriptor.get("task"), descriptor.get("profile"), 0, descriptor.get("id"))
    if descriptor != canonical:
        raise ValueError("descriptor must match the canonical catalog entry exactly")
    return canonical


def case(descriptor: dict, variant: int = 0) -> dict:
    """Describe actual shapes and numerical parameters, including stress cases.

    Development variants 0/4/5 and final variants 6/10/11 have disjoint shapes
    within every descriptor. Variants 1/2/3 are *unscored* zero/large/tiny
    boundary fixtures; all other variants use the descriptor's distribution.
    """
    descriptor = _canonical(descriptor)
    variant = _variant(variant)
    task = descriptor["task"]
    rows, width = descriptor["shapes"][0]
    if variant:
        rows += variant
        width += 2 * variant - 1
    specs = _specs(task, _shapes(task, rows, width))
    distribution = descriptor["distribution"]
    params = dict(descriptor["parameters"])
    if variant in (1, 2, 3):
        distribution = {1: "zero", 2: "large_scale", 3: "tiny_scale"}[variant]
        params = _parameters(task, "ordinary")
        params["scale"] *= {1: 0.0, 2: 1e4, 3: 1e-6}[variant]
    return {
        "schema_version": 1,
        "task": task,
        "profile": descriptor["profile"],
        "difficulty_id": descriptor["id"],
        "variant": variant,
        "stage": "unscored_stress" if variant in (1, 2, 3) else "scored",
        "shapes": [item["shape"].copy() for item in specs],
        "inputs": specs,
        "distribution": distribution,
        "parameters": params,
        "input_bytes": _bytes(specs),
    }


def inputs(descriptor: dict, seed: int, variant: int = 0) -> tuple[torch.Tensor, ...]:
    """Generate fresh float32 (and target int64) tensors on CPU with a local RNG."""
    seed = _seed(seed)
    metadata = case(descriptor, variant)
    task = metadata["task"]
    params = metadata["parameters"]
    dimensions = metadata["shapes"]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def matrix(shape: list[int]) -> torch.Tensor:
        value = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
        value.mul_(params["scale"])
        # Slice scaling avoids additional full-sized temporary tensors.
        for index, scale in enumerate(params["row_scales"]):
            value[index::len(params["row_scales"])].mul_(scale)
        for index, scale in enumerate(params["column_scales"]):
            value[:, index::len(params["column_scales"])].mul_(scale)
        value.add_(params["offset"])
        return value

    x = matrix(dimensions[0])
    if task == "rmsnorm":
        w = torch.randn(dimensions[1], generator=gen, device="cpu", dtype=torch.float32)
        w.mul_(params["weight_scale"]).add_(1.0)
        if params["alternating_weight_sign"]:
            w[::2].neg_()
        return x, w
    if task == "swiglu":
        return x, matrix(dimensions[1])
    if task == "cross_entropy":
        targets = torch.randint(dimensions[0][-1], dimensions[1], generator=gen, device="cpu", dtype=torch.int64)
        return x, targets
    return (x,)
