"""Behavioral tests for the correctness-gated kernel environment."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import pytest

from kernel_env import KernelEnvironment, evaluate


ROOT = Path(__file__).resolve().parents[1]
STARTERS = ROOT / "starters"
TASKS = (
    "rmsnorm",
    "softmax",
    "swiglu",
    "cross_entropy",
    "attention",
    "newton_schulz",
)


def _record(solution: Path, task: str, **kwargs) -> dict:
    return evaluate(
        solution,
        task,
        device="cpu",
        profile="smoke",
        phase="dev",
        trials=1,
        timeout=30,
        **kwargs,
    )


def _write_candidate(tmp_path: Path, source: str, name: str = "solution.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _metric(record: dict, *names: str):
    """Find a named measurement in the normalized metrics object."""
    metrics = record.get("metrics", {})
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


@pytest.mark.parametrize("task", TASKS)
def test_all_starters_pass_cpu_smoke(task: str, tmp_path: Path) -> None:
    """Every shipped task has a runnable, finite CPU smoke baseline."""
    result = _record(STARTERS / f"{task}.py", task, output_dir=tmp_path / "records")

    assert result["status"] == "ok", result
    assert result["reward"] is not None and math.isfinite(float(result["reward"]))
    assert float(result["reward"]) > 0
    latency = _metric(result, "latency_ms", "median_ms", "candidate_ms", "time_ms")
    assert latency is not None and math.isfinite(float(latency)) and float(latency) > 0
    assert result.get("metrics")
    assert result.get("provenance")


def test_invalid_nan_output_has_zero_reward(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "import torch\ndef kernel(x):\n    return torch.full_like(x, float('nan'))\n",
    )
    result = _record(candidate, "softmax")

    assert result["status"] == "invalid"
    assert result["reward"] == 0


def test_invalid_shape_and_dtype_are_rejected(tmp_path: Path) -> None:
    wrong_shape = _write_candidate(
        tmp_path,
        "def kernel(x):\n    return x[..., :-1]\n",
        "wrong_shape.py",
    )
    wrong_dtype = _write_candidate(
        tmp_path,
        "def kernel(x):\n    return x.to(dtype=__import__('torch').float64)\n",
        "wrong_dtype.py",
    )

    assert _record(wrong_shape, "softmax")["status"] == "invalid"
    assert _record(wrong_dtype, "softmax")["status"] == "invalid"


def test_input_mutation_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "import torch\ndef kernel(x):\n    x.mul_(2)\n    return torch.softmax(x, dim=-1)\n",
    )
    result = _record(candidate, "softmax")

    assert result["status"] == "invalid"
    assert result["reward"] == 0


def test_cached_first_output_is_rejected_on_fresh_input(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "import torch\n_cached = None\ndef kernel(x):\n"
        "    global _cached\n"
        "    if _cached is None:\n        _cached = torch.softmax(x, dim=-1).clone()\n"
        "    return _cached\n",
    )
    result = evaluate(
        candidate,
        "softmax",
        device="cpu",
        profile="smoke",
        phase="dev",
        trials=2,
        timeout=30,
    )

    assert result["status"] == "invalid"
    assert result["reward"] == 0


def test_corruption_after_warmups_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "import torch\n_calls = 0\ndef kernel(x):\n"
        "    global _calls\n    _calls += 1\n"
        "    y = torch.softmax(x, dim=-1)\n"
        "    return y if _calls <= 2 else torch.zeros_like(y)\n",
    )
    result = _record(candidate, "softmax")

    assert result["status"] == "invalid"
    assert result["reward"] == 0


def test_hung_candidate_times_out_with_bounded_wall_time(tmp_path: Path) -> None:
    candidate = _write_candidate(
        tmp_path,
        "def kernel(x):\n    while True:\n        pass\n",
    )
    started = time.monotonic()
    result = evaluate(
        candidate,
        "softmax",
        device="cpu",
        profile="smoke",
        phase="dev",
        trials=1,
        timeout=3,
    )
    elapsed = time.monotonic() - started

    assert result["status"] == "error"
    assert result["reward"] is None
    assert elapsed < 12


def test_standard_profile_without_calibration_is_blocked_but_keeps_correctness(
    tmp_path: Path, monkeypatch,
) -> None:
    # Test standard reward policy with a reduced numerical fixture; a loaded
    # machine must not turn this gate regression into a 64-MiB timing run.
    import kernel_env.environment as environment_module
    original_shapes = environment_module.tasks.shapes
    monkeypatch.setattr(environment_module.tasks, "shapes",
                        lambda name, profile: ((64, 257),) if name == "softmax" else original_shapes(name, profile))
    monkeypatch.setattr(environment_module, "_load",
                        lambda: {"active_processes": [], "heavy_processes": [], "fixture": "simulated_idle"})
    result = evaluate(
        STARTERS / "softmax.py",
        "softmax",
        device="cpu",
        profile="standard",
        phase="dev",
        trials=1,
        timeout=45,
    )

    assert result["status"] == "blocked"
    assert result["reward"] is None
    assert "requires a fixed idle calibration" in result["metrics"]["error"]
    assert result.get("metrics")
    assert _metric(result, "correct", "correctness_passed", "max_relative_error") is not None


def test_snapshot_and_source_hash_are_saved(tmp_path: Path) -> None:
    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")
    candidate = _write_candidate(tmp_path, source)
    output_dir = tmp_path / "saved"
    result = _record(candidate, "softmax", output_dir=output_dir)

    assert result["status"] == "ok"
    expected_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    provenance = result["provenance"]
    serialized = json.dumps(provenance, sort_keys=True)
    assert expected_hash in serialized
    assert output_dir.exists() and any(p.is_file() for p in output_dir.rglob("*"))


def test_interactive_environment_reset_step_and_final(tmp_path: Path) -> None:
    env = KernelEnvironment(
        "softmax", workspace=tmp_path / "workspace", device="cpu", profile="smoke", seed=7
    )
    observation = env.reset()
    assert isinstance(observation, dict)
    candidate_path = observation.get("candidate")
    assert candidate_path is not None

    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")
    test_transition = env.step({"type": "test", "code": source})
    assert test_transition["reward"] == 0
    assert not test_transition["terminated"] and not test_transition["truncated"]
    assert test_transition["info"]["status"] == "ok"

    final = env.final()
    assert final["terminated"] and not final["truncated"]
    assert final["reward"] is not None
    assert final["info"]["provenance"]["phase"] == "final"


def test_max_steps_one_submits_and_returns_terminal_reward(tmp_path: Path) -> None:
    env = KernelEnvironment(
        "softmax",
        workspace=tmp_path / "workspace",
        device="cpu",
        profile="smoke",
        seed=11,
        max_steps=1,
    )
    env.reset()
    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")

    transition = env.step({"type": "test", "code": source})

    assert transition["terminated"] and not transition["truncated"]
    assert transition["reward"] == transition["info"]["reward"]
    assert transition["info"]["provenance"]["phase"] == "final"


@pytest.mark.parametrize("task", ("attention", "newton_schulz"))
def test_eval_tasks_cannot_be_assigned_to_training_split(task: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train/eval split"):
        KernelEnvironment(task, workspace=tmp_path / task, split="train")


def test_reset_starts_new_episode_and_preserves_prior_evidence(tmp_path: Path) -> None:
    env = KernelEnvironment(
        "softmax", workspace=tmp_path / "workspace", device="cpu", profile="smoke", seed=19
    )
    first = env.reset()
    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")
    transition = env.step({"type": "test", "code": source})
    first_evidence = Path(first["candidate"]).parent / "evidence"
    assert transition["info"]["status"] == "ok"
    assert first_evidence.exists()
    assert any(first_evidence.rglob("record.json"))

    second = env.reset()

    assert Path(second["candidate"]).parent != Path(first["candidate"]).parent
    assert Path(first["candidate"]).exists()
    assert any(first_evidence.rglob("record.json"))


def test_invalid_final_candidate_terminates_with_zero_reward(tmp_path: Path) -> None:
    env = KernelEnvironment(
        "softmax", workspace=tmp_path / "workspace", device="cpu", profile="smoke"
    )
    env.reset()
    invalid = "import torch\ndef kernel(x):\n    return torch.full_like(x, float('nan'))\n"
    env.step({"type": "edit", "code": invalid})

    transition = env.step({"type": "submit"})

    assert transition["terminated"] and not transition["truncated"]
    assert transition["reward"] == 0
    assert transition["info"]["status"] == "invalid"
    assert "speedup" not in transition["info"]["metrics"]


def test_unavailable_cuda_truncates_with_null_reward(tmp_path: Path, monkeypatch) -> None:
    # This is a device-unavailability routing test, including on CUDA hosts;
    # it must never launch an accelerator workload as part of CPU smoke QA.
    import kernel_env.environment as environment_module
    monkeypatch.setattr(environment_module.torch.cuda, "is_available", lambda: False)
    env = KernelEnvironment(
        "softmax", workspace=tmp_path / "workspace", device="cuda", profile="smoke"
    )
    env.reset()
    transition = env.step({"type": "submit"})

    assert transition["truncated"] and not transition["terminated"]
    assert transition["reward"] is None
    assert transition["info"]["status"] == "blocked"


def test_episode_wall_budget_truncates_without_sleep(tmp_path: Path, monkeypatch) -> None:
    import kernel_env.environment as environment_module

    env = KernelEnvironment(
        "softmax",
        workspace=tmp_path / "workspace",
        device="cpu",
        profile="smoke",
        max_episode_seconds=1,
    )
    env.reset()
    monkeypatch.setattr(environment_module.time, "monotonic", lambda: env.started + 2)

    transition = env.step({"type": "test"})

    assert transition["truncated"] and not transition["terminated"]
    assert transition["reward"] is None
    assert transition["info"]["status"] == "blocked"


@pytest.mark.parametrize("seconds", (float("nan"), float("inf"), float("-inf")))
def test_episode_budget_must_be_finite(seconds: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_episode_seconds"):
        KernelEnvironment(
            "softmax", workspace=tmp_path / "workspace", max_episode_seconds=seconds
        )


def test_accelerator_lock_timeout_is_blocked_with_null_reward(
    tmp_path: Path, monkeypatch
) -> None:
    import contextlib

    import kernel_env.environment as environment_module
    from suite.records import make_record
    from suite.resources import AcceleratorBusy

    @contextlib.contextmanager
    def busy_lease():
        raise AcceleratorBusy("test lock is already held")
        yield

    def busy_lock(*, device: str, timeout_seconds: float):
        return busy_lease()

    monkeypatch.setattr(environment_module, "_check_device", lambda _device: None)
    monkeypatch.setattr(
        environment_module, "_suite", lambda: (make_record, busy_lock)
    )
    result = evaluate(
        STARTERS / "softmax.py",
        "softmax",
        device="mps",
        profile="smoke",
        phase="dev",
        trials=1,
        timeout=10,
        output_dir=tmp_path / "records",
        lock_timeout=0,
    )

    assert result["status"] == "blocked"
    assert result["reward"] is None
    assert "AcceleratorBusy" in result["metrics"]["error"]


def test_relative_output_dir_is_resolved_after_chdir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _record(STARTERS / "softmax.py", "softmax", output_dir="records")

    assert result["status"] == "ok", result
    assert any((tmp_path / "records").rglob("record.json"))


def test_worker_protocol_read_obeys_absolute_deadline() -> None:
    import socket
    import time

    from kernel_env.worker import receive

    receiver, sender = socket.socketpair()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out|deadline"):
            receive(receiver, deadline=started + 0.05)
        assert time.monotonic() - started < 1
    finally:
        receiver.close()
        sender.close()
