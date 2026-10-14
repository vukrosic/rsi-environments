import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from suite.records import loss_reward, make_record, speed_reward, validate_record, write_record
from suite.resources import accelerator_lock


def valid_record(**changes):
    fields = dict(environment="kernel", task="attention", profile="standard", status="ok",
                  reward=0.5, metrics={"speedup": 1.0}, provenance={"device": "cpu"})
    fields.update(changes)
    return make_record(**fields)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -0.1, 1.1, True, None])
def test_reject_invalid_rewards(reward):
    with pytest.raises(ValueError):
        valid_record(reward=reward)


def test_infrastructure_errors_cannot_be_training_rewards():
    assert valid_record(status="blocked", reward=None)["reward"] is None
    assert valid_record(status="error", reward=None)["reward"] is None
    with pytest.raises(ValueError):
        valid_record(status="blocked", reward=0)
    with pytest.raises(ValueError):
        valid_record(status="invalid", reward=0.5)
    assert valid_record(status="invalid", reward=0)["reward"] == 0


def test_nested_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        valid_record(metrics={"times": [1, float("nan")]})


def test_atomic_roundtrip_and_detached_records(tmp_path):
    metrics = {"speedup": 1.0}
    record = valid_record(metrics=metrics)
    metrics["speedup"] = 99
    path = tmp_path / "run.json"
    write_record(path, record)
    assert validate_record(json.loads(path.read_text()))["metrics"]["speedup"] == 1
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError):
        write_record(path, {**record, "reward": float("nan")})
    assert json.loads(path.read_text())["reward"] == 0.5


def test_reward_transforms():
    assert speed_reward(1) == 0.5
    assert speed_reward(2) > speed_reward(1)
    assert loss_reward(2) > loss_reward(3)
    assert loss_reward(0) == 1
    for value in [float("nan"), float("inf"), -1, True]:
        with pytest.raises(ValueError):
            loss_reward(value)
        with pytest.raises(ValueError):
            speed_reward(value)


def child_attempt(path):
    code = "from suite.resources import accelerator_lock,AcceleratorBusy\nimport sys\ntry:\n with accelerator_lock(path=sys.argv[1],timeout_seconds=0): pass\nexcept AcceleratorBusy:\n sys.exit(23)\n"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    return subprocess.run([sys.executable, "-c", code, str(path)], env=env, timeout=5).returncode


def test_lock_serializes_processes_and_releases_after_failure(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError):
        with accelerator_lock(path=path, timeout_seconds=0):
            assert child_attempt(path) == 23
            raise RuntimeError("candidate failed")
    assert child_attempt(path) == 0
    assert path.exists()  # unlinking a lock file lets competing processes bypass it


def test_cpu_does_not_block_on_gpu_lease(tmp_path):
    path = tmp_path / "gpu.lock"
    with accelerator_lock(path=path, timeout_seconds=0):
        with accelerator_lock(device="cpu", path=path, timeout_seconds=0) as lease:
            assert lease["accelerator_lease"] is False


def test_crashed_process_releases_lease(tmp_path):
    path = tmp_path / "gpu.lock"
    code = "from suite.resources import accelerator_lock\nimport os,sys\nwith accelerator_lock(path=sys.argv[1],timeout_seconds=0): os._exit(17)\n"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, "-c", code, str(path)], env=env, timeout=5)
    assert result.returncode == 17
    assert child_attempt(path) == 0


def test_invalid_lock_configuration_rejected(tmp_path):
    for timeout in [float("nan"), float("inf"), -1, True]:
        with pytest.raises(ValueError):
            with accelerator_lock(path=tmp_path / "gpu.lock", timeout_seconds=timeout):
                pass
    with pytest.raises(ValueError):
        with accelerator_lock(path="relative.lock"):
            pass
