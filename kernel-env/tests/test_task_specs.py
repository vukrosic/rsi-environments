"""Public task specifications are complete, stable JSON data."""

from __future__ import annotations

import json
import math

import pytest

from kernel_env import tasks


EXPECTED = {
    "rmsnorm": ("kernel(x, w) -> output", ("x", "w"), "torch.nn.functional.rms_norm"),
    "softmax": ("kernel(x) -> output", ("x",), "torch.softmax"),
    "swiglu": ("kernel(a, b) -> output", ("a", "b"), "torch.nn.functional.silu * multiply"),
    "cross_entropy": ("kernel(logits, targets) -> output", ("logits", "targets"), "torch.nn.functional.cross_entropy"),
    "attention": ("kernel(q, k, v) -> output", ("q", "k", "v"), "torch.nn.functional.scaled_dot_product_attention"),
    "newton_schulz": ("kernel(g) -> output", ("g",), "batched PyTorch matrix operations (five Newton-Schulz steps)"),
}


@pytest.mark.parametrize("name", tasks.NAMES)
def test_describe_returns_complete_json_spec_for_each_task(name: str) -> None:
    spec = tasks.describe(name, "smoke")

    # Assert the public fields individually so missing or misnamed data fails
    # with a direct indication of which contract was lost.
    assert spec["task"] == name
    assert spec["profile"] == "smoke"
    assert spec["signature"] == EXPECTED[name][0]
    assert isinstance(spec["operation"], str) and spec["operation"].strip()
    assert spec["shape_note"]
    assert [item["name"] for item in spec["inputs"]] == list(EXPECTED[name][1])
    assert all(isinstance(item["shape"], list) for item in spec["inputs"])
    assert all(item["dtype"] == "float32" for item in spec["inputs"] if item["name"] != "targets")
    assert spec["baseline"] == EXPECTED[name][2]
    assert isinstance(spec["output"]["shape"], list)
    assert spec["output"]["dtype"] == "float32"
    assert math.isfinite(spec["tolerance"]) and spec["tolerance"] > 0
    assert "max(abs(actual - reference) / max(1, abs(reference)))" in spec["error_definition"]
    assert "<= tolerance" in spec["error_definition"]
    assert isinstance(spec["constants"], dict)
    json.dumps(spec, allow_nan=False)


@pytest.mark.parametrize("name", tasks.NAMES)
@pytest.mark.parametrize("profile", ("smoke", "standard"))
def test_spec_shapes_and_tolerances_match_nominal_task_contract(name: str, profile: str) -> None:
    spec = tasks.describe(name, profile)
    shapes = tasks.shapes(name, profile)
    assert [tuple(item["shape"]) for item in spec["inputs"]] == list(shapes)
    assert spec["tolerance"] == tasks.tolerance(name)
    if name == "cross_entropy":
        assert spec["inputs"][1]["dtype"] == "int64"
        assert spec["output"]["shape"] == []
    if name == "attention":
        assert spec["output"]["shape"] == [*shapes[0][:-1], shapes[2][-1]]


def test_rmsnorm_and_newton_schulz_expose_exact_constants() -> None:
    rms = tasks.describe("rmsnorm", "smoke")
    assert rms["constants"]["eps"] == 1e-6
    assert rms["constants"]["reduction_axis"] == "last"

    ns = tasks.describe("newton_schulz", "smoke")
    assert ns["constants"]["iterations"] == 5
    assert ns["constants"]["norm_eps"] == 1e-7
    assert ns["constants"]["polynomial_coefficients"] == [3.4445, -4.7750, 2.0315]
    assert ns["constants"]["transpose_condition"] == "rows > columns"


def test_attention_and_cross_entropy_expose_exact_semantics() -> None:
    attention = tasks.describe("attention", "smoke")
    assert attention["constants"] == {
        "scale": "1 / sqrt(query_head_dim)",
        "causal": True,
        "dropout": 0.0,
    }
    assert "mask future key positions" in attention["operation"]

    cross_entropy = tasks.describe("cross_entropy", "smoke")
    assert cross_entropy["constants"] == {"reduction": "mean", "label_smoothing": 0.0}
    assert "average over the batch" in cross_entropy["operation"]


@pytest.mark.parametrize("name", ("unknown", ""))
def test_describe_rejects_unknown_task(name: str) -> None:
    with pytest.raises(ValueError, match="unknown kernel task"):
        tasks.describe(name, "smoke")


@pytest.mark.parametrize("profile", ("unknown", "", "large"))
def test_describe_rejects_unknown_profile(profile: str) -> None:
    with pytest.raises(ValueError, match="profile must be"):
        tasks.describe("softmax", profile)
