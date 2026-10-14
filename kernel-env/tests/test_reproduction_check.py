import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import reproduction_check as check


class FakeEnvironment:
    instances = []
    misreport_invalid_reward = False

    def __init__(self, task, workspace, **kwargs):
        self.task = task
        self.workspace = Path(workspace)
        self.kwargs = kwargs
        self.episode_index = -1
        self.done = True
        self.instances.append(self)

    def reset(self, *, seed=None):
        self.episode_index += 1
        self.active_seed = seed
        self.done = False
        self.episode_dir = self.workspace / "episodes" / f"episode-{self.episode_index}-{len(self.instances)}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.candidate = self.episode_dir / "candidate.py"
        self.candidate.write_text("def kernel(x): return x\n")
        difficulty = self.kwargs["difficulty_id"]
        return {
            "seed": seed,
            "episode_index": self.episode_index,
            "difficulty": {"id": difficulty},
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("episode ended or was not reset; call reset()")
        if action.get("code") is not None:
            self.candidate.write_text(action["code"])
        status = "ok"
        reward = 0.0
        metrics = {"correctness_passed": True}
        terminated = action["type"] == "submit"
        if terminated:
            if "zeros_like" in self.candidate.read_text():
                status = "invalid"
                reward = 0.1 if self.misreport_invalid_reward else 0.0
                metrics = {"error": "softmax row normalization error 1 exceeds 0.002"}
            else:
                reward = 0.25
            self.done = True
        info = {
            "status": status,
            "metrics": metrics,
            "exclude_from_rl": True,
            "rl_exclusion_reasons": ["smoke_profile"],
        }
        transition = {
            "info": info,
            "reward": reward,
            "terminated": terminated,
            "truncated": False,
        }
        index = len(list(self.episode_dir.glob("transition-*.json")))
        (self.episode_dir / f"transition-{index:02d}.json").write_text(json.dumps(transition))
        run_dir = self.episode_dir / "evidence" / f"run-{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "record.json").write_text(json.dumps({"status": status, "reward": reward, "metrics": metrics}))
        return transition


def test_mocked_reproduction_receipt_binds_runtime_and_cpu_execution(tmp_path: Path) -> None:
    FakeEnvironment.instances = []
    FakeEnvironment.misreport_invalid_reward = False
    output = tmp_path / "receipt.json"
    workspace = tmp_path / "workspace"

    receipt = check.run_reproduction(output, workspace, environment_class=FakeEnvironment)

    assert receipt["status"] == "ok"
    assert receipt["scope"] == "local CPU smoke compatibility only"
    assert receipt["execution"]["device"] == "cpu"
    assert receipt["execution"]["accelerator_requested"] is False
    assert receipt["execution"]["evaluation_timeout_seconds"] == 30
    assert receipt["execution"]["episode_budget_seconds"] == 90
    assert all(item.kwargs["device"] == "cpu" for item in FakeEnvironment.instances)
    assert all(item.kwargs["timeout"] == 30 for item in FakeEnvironment.instances)
    assert all(item.kwargs["max_episode_seconds"] == 90 for item in FakeEnvironment.instances)
    assert receipt["limitations"] == {
        "hardware_timing_validated": False,
        "outer_weights_trained": False,
        "external_adoption_established": False,
    }
    runtime = receipt["runtime"]
    assert runtime["python_version"]
    assert runtime["platform"]
    assert runtime["torch_version"]
    assert runtime["numpy_version"]
    assert runtime["package"]["version"]
    assert runtime["package"]["origin"]
    assert runtime["package"]["origin_kind"] in {"source_checkout", "installed_package"}
    assert runtime["package"]["source_sha256"]
    assert receipt["checks"]["nominal_ordinary_seed23_test_and_final"] == "passed"
    assert receipt["checks"]["wide_ordinary_zero_output_rejected_for_row_mass"] == "passed"
    assert receipt["checks"]["reset_preserves_prior_artifacts"] == "passed"
    assert receipt["full_records"]["path"] == "full-records.json"
    full_records = json.loads((workspace / "full-records.json").read_text())
    assert full_records["transitions"]
    assert full_records["score_records"]
    assert "full-records.json" in receipt["workspace_artifacts"]
    assert json.loads(output.read_text()) == receipt


def test_misreported_invalid_score_becomes_saved_error_receipt(tmp_path: Path) -> None:
    FakeEnvironment.instances = []
    FakeEnvironment.misreport_invalid_reward = True
    output = tmp_path / "error.json"
    workspace = tmp_path / "error-workspace"

    receipt = check.run_reproduction(output, workspace, environment_class=FakeEnvironment)

    assert receipt["status"] == "error"
    assert "expected reward 0.0, got 0.1" in receipt["error"]
    assert json.loads(output.read_text())["status"] == "error"
    assert (workspace / "full-records.json").is_file()


def test_existing_output_or_workspace_is_refused_without_changes(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("keep receipt\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("keep workspace\n")

    with pytest.raises(FileExistsError, match="receipt"):
        check.run_reproduction(output, tmp_path / "new-workspace", environment_class=FakeEnvironment)
    with pytest.raises(FileExistsError, match="workspace"):
        check.run_reproduction(tmp_path / "new-receipt.json", workspace, environment_class=FakeEnvironment)
    assert output.read_text() == "keep receipt\n"
    assert (workspace / "keep.txt").read_text() == "keep workspace\n"


def test_adjacent_source_version_overrides_stale_installed_metadata(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    package_dir = source_root / "kernel_env"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("# source package\n")
    (source_root / "pyproject.toml").write_text('[project]\nversion = "9.8.7"\n')
    stale_distribution = SimpleNamespace(
        version="1.0.0",
        locate_file=lambda _name: tmp_path / "stale-install" / "kernel_env" / "__init__.py",
    )
    monkeypatch.setattr(check.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        check.importlib, "import_module", lambda _name: SimpleNamespace(__file__=str(package_file))
    )
    monkeypatch.setattr(
        check.importlib.metadata, "distribution", lambda _name: stale_distribution
    )

    package = check._package_details()

    assert package["version"] == "9.8.7"
    assert package["version_source"] == "adjacent source pyproject.toml"
    assert package["origin_kind"] == "source_checkout"


def test_mismatched_installed_metadata_is_reported_as_unknown(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "site-packages" / "kernel_env"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("# installed package\n")
    mismatched_distribution = SimpleNamespace(
        version="2.3.4",
        locate_file=lambda _name: tmp_path / "other-site" / "kernel_env" / "__init__.py",
    )
    monkeypatch.setattr(check.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        check.importlib, "import_module", lambda _name: SimpleNamespace(__file__=str(package_file))
    )
    monkeypatch.setattr(
        check.importlib.metadata, "distribution", lambda _name: mismatched_distribution
    )

    package = check._package_details()

    assert package["version"] == "unknown"
    assert package["version_source"] == "unknown; installed distribution path mismatch"
    assert package["origin_kind"] == "installed_package"
