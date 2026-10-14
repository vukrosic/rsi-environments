"""Changed sampler integration and infrastructure gates; CPU or bounded mocks."""
import json
from pathlib import Path

import pytest
import numpy as np

from kernel_env import KernelEnvironment, evaluate
from kernel_env import difficulty
import kernel_env.environment as envmod

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task,case_id", [
    ("rmsnorm", "odd/offset"), ("softmax", "wide/offset"),
    ("swiglu", "wide/mixed_scale"), ("cross_entropy", "nominal/mixed_scale"),
])
def test_real_cpu_difficulty_score_preserves_actual_case(task, case_id, tmp_path):
    result = evaluate(ROOT / "starters" / f"{task}.py", task, profile="smoke",
                      seed=19, trials=1, difficulty_id=case_id, output_dir=tmp_path)
    assert result["status"] == "ok", result
    descriptor = result["provenance"]["difficulty"]
    assert descriptor == difficulty.choose(task, "smoke", 19, case_id)
    row = result["metrics"]["trials"][0]
    actual = difficulty.case(descriptor, row["variant"])
    assert row["case"] == dict(actual, seed=row["seed"])
    assert row["shapes"] == actual["shapes"]
    assert row["baseline_after_case"]["seed"] != row["case"]["seed"]
    assert not result["trainable"] and result["exclude_from_rl"]


def test_training_reset_sampling_and_explicit_seed_reproduce(tmp_path):
    env = KernelEnvironment("softmax", tmp_path / "train", seed=19)
    first = env.reset()
    second = env.reset()
    replay = env.reset(seed=19)
    assert (first["seed"], second["seed"], replay["seed"]) == (19, 20, 19)
    assert first["difficulty"] == replay["difficulty"]
    assert first["difficulty"] == difficulty.choose("softmax", "smoke", 19)
    assert first["candidate"] != replay["candidate"]


def test_observation_nominal_shapes_match_selected_difficulty(tmp_path):
    env = KernelEnvironment("softmax", tmp_path, profile="standard", difficulty_id="wide/offset")
    observation = env.reset()
    assert observation["task_spec"]["inputs"] == observation["difficulty"]["inputs"]
    assert observation["task_spec"]["output"]["shape"] == observation["difficulty"]["shapes"][0]


@pytest.mark.parametrize("task", ("attention", "newton_schulz"))
def test_eval_reset_never_calls_training_sampler(task, tmp_path, monkeypatch):
    monkeypatch.setattr(difficulty, "choose", lambda *a, **k: pytest.fail("heldout sampler changed"))
    env = KernelEnvironment(task, tmp_path / task, seed=19)
    first, second = env.reset(), env.reset()
    assert first["seed"] == second["seed"] == 19
    assert first["difficulty"] == {"id": "legacy", "sampling": "frozen_eval"}


def _receipt(monkeypatch):
    system = {"device": "cpu", "fixture": True}
    monkeypatch.setattr(envmod, "_system", lambda _device: system)
    return {"status": "ok", "task": "softmax", "system": system,
            "tasks_sha256": envmod._hash(envmod.tasks.__file__),
            "scorer_sha256": envmod._integrity(), "difficulty": {"id": "legacy"},
            "baseline_ms_by_variant": {str(v): 1.0 for v in (0, 4, 5, 6, 10, 11)}}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True, "1", 10**1000])
def test_calibration_rejects_nonfinite_or_invalid_idle_times(value, tmp_path, monkeypatch):
    receipt = _receipt(monkeypatch)
    receipt["baseline_ms_by_variant"]["0"] = value
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(envmod.Blocked, match="invalid calibration receipt"):
        envmod._calibration_receipt(path, "softmax", "cpu", None, envmod._integrity())


def test_calibration_rejects_wrong_difficulty_and_missing_variants(tmp_path, monkeypatch):
    receipt = _receipt(monkeypatch)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt))
    selected = difficulty.choose("softmax", "standard", 19)
    with pytest.raises(envmod.Blocked, match="difficulty differs"):
        envmod._calibration_receipt(path, "softmax", "cpu", selected, envmod._integrity())
    del receipt["baseline_ms_by_variant"]["11"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(envmod.Blocked, match="all six"):
        envmod._calibration_receipt(path, "softmax", "cpu", None, envmod._integrity())


def test_trusted_baseline_protocol_error_is_not_candidate_failure():
    class BrokenBaseline:
        def call(self, arrays):
            raise envmod.InvalidCandidate("wrong comparator dtype")
    with pytest.raises(envmod.Blocked, match="baseline failed output protocol"):
        envmod._baseline_call(BrokenBaseline(), ())


def _mock_measurement(monkeypatch):
    rows = [{"variant": 0, "candidate_ms": 1.0, "baseline_ms": 1.0,
             "baseline_before_ms": 1.0, "baseline_after_ms": 1.0,
             "speedup": 1.0, "error": 0.0, "mocked": True}]
    monkeypatch.setattr(envmod, "_run_cases", lambda *a, **k: rows)
    monkeypatch.setattr(envmod, "_load", lambda: {"active_processes": [], "heavy_processes": []})


def test_trusted_scorer_drift_excludes_standard_train_reward(tmp_path, monkeypatch):
    _mock_measurement(monkeypatch)
    baseline = envmod._integrity()
    calls = iter((baseline, dict(baseline, drift="fixture")))
    monkeypatch.setattr(envmod, "_integrity", lambda: next(calls))
    result = evaluate(ROOT / "starters" / "softmax.py", "softmax", profile="standard",
                      phase="final", output_dir=tmp_path)
    assert result["status"] == "blocked" and result["reward"] is None
    assert not result["trainable"] and result["exclude_from_rl"]
    assert "trusted scorer source changed" in result["metrics"]["error"]


def test_failed_final_calibration_integrity_check_cannot_leave_status_ok(tmp_path, monkeypatch):
    rows = [{"variant": v, "baseline_before_ms": 1.0} for _ in range(3) for v in (0, 4, 5, 6, 10, 11)]
    monkeypatch.setattr(envmod, "_run_cases", lambda *a, **k: rows)
    monkeypatch.setattr(envmod, "_load", lambda: {"active_processes": [], "heavy_processes": []})
    original = envmod._integrity()
    calls = iter((original, dict(original, drift="fixture")))
    monkeypatch.setattr(envmod, "_integrity", lambda: next(calls))
    result = envmod.calibrate("softmax", device="cpu", output_dir=tmp_path)
    assert result["status"] == "blocked"
    assert "source changed" in result["error"]


@pytest.mark.parametrize("kind", ("missing_mass", "zero_mass", "negative_probability"))
def test_softmax_invariants_reject_small_elementwise_but_wrong_probabilities(kind):
    arrays = (np.zeros((2, 1024), dtype=np.float32),)
    expected = np.full((2, 1024), 1 / 1024, dtype=np.float32)
    actual = expected.copy()
    if kind == "missing_mass":
        actual *= 0.99
    elif kind == "zero_mass":
        actual.fill(0)
    else:
        actual[:, 0] = -0.0001
    assert np.abs(actual - expected).max() < 2e-3
    metadata = {"device": "cpu", "input_hashes": [envmod.fingerprint(arrays[0])]}
    with pytest.raises(envmod.InvalidCandidate, match="softmax"):
        envmod._validate(actual, expected, metadata, arrays, "cpu", 2e-3, task="softmax")


def test_explicit_malformed_submit_seals_episode_without_kernel_score(tmp_path):
    env = KernelEnvironment("softmax", tmp_path, profile="standard", max_steps=5)
    env.reset()
    transition = env.step({"type": "submit", "code": ""})
    assert transition["terminated"] and not transition["truncated"]
    assert transition["reward"] == 0 and transition["trainable"]
    assert transition["info"]["provenance"]["stage"] == "policy_schema"
    assert transition["info"]["metrics"]["scored"] is False
    with pytest.raises(RuntimeError, match="episode ended"):
        env.final()
