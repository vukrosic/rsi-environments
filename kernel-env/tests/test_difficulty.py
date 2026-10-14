"""Bounded CPU checks for training-only shape and distribution sampling."""

from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from kernel_env import difficulty, tasks


VARIANTS = tuple(range(difficulty.MAX_VARIANT + 1))
DEV = (0, 4, 5)
FINAL = (6, 10, 11)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def input_bytes(specs):
    return sum(math.prod(item["shape"]) * (8 if item["dtype"] == "int64" else 4) for item in specs)


@pytest.mark.parametrize("task", difficulty.TRAIN_TASKS)
@pytest.mark.parametrize("profile", difficulty.PROFILES)
def test_catalog_is_stable_json_and_metadata_is_bounded(task, profile):
    descriptors = difficulty.catalog(task, profile)
    assert descriptors == difficulty.catalog(task, profile)
    assert len(descriptors) == 9
    assert len({entry["id"] for entry in descriptors}) == len(descriptors)
    nominal_bytes = input_bytes(tasks.describe(task, profile)["inputs"])
    for descriptor in descriptors:
        json.dumps(descriptor, allow_nan=False)
        assert descriptor["schema_version"] == 1
        assert descriptor["task"] == task
        assert descriptor["profile"] == profile
        assert descriptor["input_names"] == [item["name"] for item in descriptor["inputs"]]
        assert descriptor["dtypes"] == [item["dtype"] for item in descriptor["inputs"]]
        assert descriptor["shapes"] == [item["shape"] for item in descriptor["inputs"]]
        for variant in VARIANTS:
            metadata = difficulty.case(descriptor, variant)
            json.dumps(metadata, allow_nan=False)
            assert metadata["input_bytes"] == input_bytes(metadata["inputs"])
            assert all(all(isinstance(n, int) and n > 0 for n in item["shape"]) for item in metadata["inputs"])
            if profile == "smoke":
                rows, width = metadata["shapes"][0]
                assert rows <= 128 and width <= 1025
            else:
                # Standard tests inspect metadata, never allocate a large fleet.
                assert metadata["input_bytes"] <= 1.2 * nominal_bytes
        dev_shapes = {tuple(map(tuple, difficulty.case(descriptor, v)["shapes"])) for v in DEV}
        final_shapes = {tuple(map(tuple, difficulty.case(descriptor, v)["shapes"])) for v in FINAL}
        assert len(dev_shapes) == len(final_shapes) == 3
        assert dev_shapes.isdisjoint(final_shapes)
        selected = difficulty.choose(task, profile, 9, descriptor["id"])
        assert selected == descriptor
        assert selected is not descriptor
    # Mutating returned JSON must not mutate the canonical catalog.
    descriptors[0]["parameters"]["row_scales"][0] = 999
    assert difficulty.catalog(task, profile)[0]["parameters"]["row_scales"] == [1.0]


@pytest.mark.parametrize("task", difficulty.TRAIN_TASKS)
def test_seed_selection_and_generator_do_not_change_global_rng(task):
    before = torch.random.get_rng_state().clone()
    selections = [difficulty.choose(task, "smoke", seed) for seed in range(30)]
    assert selections == [difficulty.choose(task, "smoke", seed) for seed in range(30)]
    assert len({descriptor["id"] for descriptor in selections}) > 1
    for descriptor in difficulty.catalog(task, "smoke"):
        for variant in VARIANTS:
            first = difficulty.inputs(descriptor, 123, variant)
            repeated = difficulty.inputs(descriptor, 123, variant)
            fresh = difficulty.inputs(descriptor, 124, variant)
            assert all(torch.equal(a, b) for a, b in zip(first, repeated))
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, repeated))
            if variant != 1:
                assert not torch.equal(first[0], fresh[0])
                assert torch.count_nonzero(first[0]) > 0
                # Outputs, not merely targets/weights, depend on sampled data.
                out1 = tasks.reference(task, *first)
                out2 = tasks.reference(task, *fresh)
                assert not torch.equal(out1, out2)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("task", difficulty.TRAIN_TASKS)
def test_nominal_ordinary_preserves_historical_variant_zero(task):
    descriptor = difficulty.choose(task, "smoke", 42, "nominal/ordinary")
    sampled = difficulty.inputs(descriptor, 42)
    historical = tasks.inputs(task, "smoke", 42)
    assert all(torch.equal(new, old) for new, old in zip(sampled, historical))


@pytest.mark.parametrize("task", difficulty.TRAIN_TASKS)
def test_every_smoke_family_variant_matches_independent_reference(task):
    for descriptor in difficulty.catalog(task, "smoke"):
        for variant in VARIANTS:
            metadata = difficulty.case(descriptor, variant)
            xs = difficulty.inputs(descriptor, 20261014, variant)
            saved = tuple(x.clone() for x in xs)
            assert [list(x.shape) for x in xs] == metadata["shapes"]
            assert all(x.device.type == "cpu" and x.is_contiguous() for x in xs)
            assert all(torch.isfinite(x).all() for x in xs)
            assert [str(x.dtype).removeprefix("torch.") for x in xs] == descriptor["dtypes"]
            if task == "cross_entropy":
                assert xs[1].min() >= 0 and xs[1].max() < xs[0].shape[-1]
            expected = tasks.reference(task, *xs)
            actual = tasks.baseline(task, *xs)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype == torch.float32
            assert actual.device.type == expected.device.type == "cpu"
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            error = ((actual - expected).abs() / expected.abs().clamp_min(1)).max().item()
            assert error <= tasks.tolerance(task), (descriptor["id"], variant, error)
            assert all(torch.equal(x, old) for x, old in zip(xs, saved))
            if variant in (1, 2, 3):
                assert metadata["stage"] == "unscored_stress"
                assert metadata["distribution"] == {1: "zero", 2: "large_scale", 3: "tiny_scale"}[variant]
                assert metadata["parameters"]["offset"] == 0
                if variant == 1:
                    assert torch.count_nonzero(xs[0]) == 0
            else:
                assert metadata["stage"] == "scored"
                assert metadata["distribution"] == descriptor["distribution"]
                assert torch.count_nonzero(xs[0]) > 0


@pytest.mark.parametrize("task", ("attention", "newton_schulz", "unknown", ""))
def test_heldout_and_unknown_tasks_rejected(task):
    with pytest.raises(ValueError, match="training tasks only"):
        difficulty.catalog(task, "smoke")
    with pytest.raises(ValueError, match="training tasks only"):
        difficulty.choose(task, "smoke", 0)
    with pytest.raises(ValueError, match="training tasks only"):
        difficulty.inputs({"task": task, "profile": "smoke", "id": "nominal/ordinary"}, 0)


@pytest.mark.parametrize("seed", (-1, 2**63, True, False, 1.0, "1", None))
def test_invalid_seeds_rejected(seed):
    descriptor = difficulty.catalog("softmax", "smoke")[0]
    with pytest.raises(ValueError, match="seed must"):
        difficulty.choose("softmax", "smoke", seed)
    with pytest.raises(ValueError, match="seed must"):
        difficulty.inputs(descriptor, seed)


@pytest.mark.parametrize("variant", (-1, 12, True, 1.0, "1", None))
def test_invalid_variants_rejected(variant):
    descriptor = difficulty.catalog("softmax", "smoke")[0]
    with pytest.raises(ValueError, match="variant must"):
        difficulty.case(descriptor, variant)
    with pytest.raises(ValueError, match="variant must"):
        difficulty.inputs(descriptor, 0, variant)


def test_invalid_ids_profiles_and_tampered_descriptors_rejected():
    with pytest.raises(ValueError, match="profile must"):
        difficulty.catalog("softmax", "huge")
    for bad_id in ("unknown", "", 4, False):
        with pytest.raises(ValueError, match="difficulty_id"):
            difficulty.choose("softmax", "smoke", 0, bad_id)
    descriptor = difficulty.choose("softmax", "smoke", difficulty.MAX_SEED)
    original = copy.deepcopy(descriptor)
    for key, value in (("shapes", [[10**9, 10**9]]), ("parameters", {"scale": 1e100}), ("schema_version", 2)):
        altered = copy.deepcopy(original)
        altered[key] = value
        with pytest.raises(ValueError, match="canonical"):
            difficulty.inputs(altered, 0)
    with pytest.raises(ValueError, match="canonical"):
        difficulty.inputs(None, 0)
    # Descriptor and case dictionaries have no shared mutable state.
    metadata = difficulty.case(descriptor, 4)
    metadata["parameters"]["row_scales"][0] = 999
    metadata["inputs"][0]["shape"][0] = 10**9
    assert descriptor == original
    assert difficulty.case(descriptor, 4)["parameters"]["row_scales"][0] != 999
