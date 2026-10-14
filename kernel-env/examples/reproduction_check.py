#!/usr/bin/env python3
"""Run bounded local CPU smoke episodes against kernel-env and save a receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import tomllib
import traceback
import uuid


EVALUATION_TIMEOUT_SECONDS = 30
EPISODE_BUDGET_SECONDS = 90
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_details() -> dict:
    if importlib.util.find_spec("kernel_env") is None and (PROJECT_ROOT / "kernel_env" / "__init__.py").is_file():
        sys.path.insert(0, str(PROJECT_ROOT))
    package = importlib.import_module("kernel_env")
    package_path = Path(package.__file__).resolve()
    package_root = package_path.parent
    source_config = package_root.parent / "pyproject.toml"
    if source_config.is_file():
        with source_config.open("rb") as config_file:
            version = tomllib.load(config_file)["project"]["version"]
        version_source = "adjacent source pyproject.toml"
        origin_kind = "source_checkout"
    else:
        try:
            distribution = importlib.metadata.distribution("rsi-kernel-env")
            distribution_package = Path(
                distribution.locate_file("kernel_env/__init__.py")
            ).resolve()
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
            version_source = "unknown; no installed distribution metadata"
            origin_kind = "installed_package"
        else:
            origin_kind = "installed_package"
            if distribution_package == package_path:
                version = distribution.version
                version_source = "matching installed distribution metadata"
            else:
                version = "unknown"
                version_source = "unknown; installed distribution path mismatch"

    source_hashes: dict[str, str] = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts or path.is_symlink():
            continue
        source_hashes[path.relative_to(package_root.parent).as_posix()] = _sha256(path)
    suite_root = package_root.parent / "suite"
    if suite_root.is_dir():
        for path in sorted(suite_root.rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            source_hashes[path.relative_to(package_root.parent).as_posix()] = _sha256(path)

    return {
        "name": "rsi-kernel-env",
        "version": version,
        "version_source": version_source,
        "origin": str(package_path),
        "origin_kind": origin_kind,
        "source_sha256": source_hashes,
    }


def _runtime_details() -> dict:
    import numpy
    import torch

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "package": _package_details(),
    }


def _validate_destinations(output: Path, workspace: Path | None) -> tuple[Path, Path]:
    output = output.expanduser().resolve(strict=False)
    if workspace is None:
        workspace = output.parent / f"{output.stem}-workspace-{uuid.uuid4().hex[:10]}"
    else:
        workspace = workspace.expanduser().resolve(strict=False)

    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")
    if os.path.lexists(workspace):
        raise FileExistsError(f"refusing to reuse existing workspace: {workspace}")
    if output == workspace or output.is_relative_to(workspace):
        raise ValueError("receipt path must be outside the new workspace")
    return output, workspace


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _status(transition: dict, expected: str, *, reward: float | None = None) -> dict:
    info = transition.get("info", {})
    _require(info.get("status") == expected, f"expected status {expected!r}, got {info.get('status')!r}")
    if reward is not None:
        _require(transition.get("reward") == reward, f"expected reward {reward}, got {transition.get('reward')!r}")
    return info


def _finite_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _artifact_hashes(workspace: Path) -> dict[str, dict]:
    return {
        path.relative_to(workspace).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(workspace.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _full_records(workspace: Path) -> dict:
    transitions = []
    score_records = []
    for path in sorted(workspace.glob("episodes/*/transition-*.json")):
        transitions.append({"path": path.relative_to(workspace).as_posix(), "record": json.loads(path.read_text())})
    for path in sorted(workspace.glob("episodes/*/evidence/*/record.json")):
        score_records.append({"path": path.relative_to(workspace).as_posix(), "record": json.loads(path.read_text())})
    return {"transitions": transitions, "score_records": score_records}


def _expect_post_final_error(environment, workspace: Path) -> None:
    before = _artifact_hashes(workspace)
    try:
        environment.step({"type": "test"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("step after final submission did not raise RuntimeError")
    _require(_artifact_hashes(workspace) == before, "post-final action changed saved artifacts")


def _environment_class():
    return importlib.import_module("kernel_env").KernelEnvironment


def _run_checks(workspace: Path, environment_class=None) -> dict:
    environment_class = environment_class or _environment_class()
    common = {
        "device": "cpu",
        "profile": "smoke",
        "split": "train",
        "timeout": EVALUATION_TIMEOUT_SECONDS,
        "max_steps": 2,
        "max_episode_seconds": EPISODE_BUDGET_SECONDS,
    }
    checks: dict[str, str] = {}

    nominal = environment_class(
        "softmax", workspace=workspace, seed=23, difficulty_id="nominal/ordinary", **common
    )
    nominal_observation = nominal.reset(seed=23)
    _require(nominal_observation["difficulty"]["id"] == "nominal/ordinary", "nominal case selection drifted")
    _require(nominal_observation["seed"] == 23, "nominal episode seed drifted")
    test_transition = nominal.step({"type": "test"})
    test_info = _status(test_transition, "ok", reward=0.0)
    _require(test_transition["terminated"] is False, "test action unexpectedly terminated the episode")
    _require(test_info.get("metrics", {}).get("correctness_passed") is True, "starter test did not pass correctness")
    final_transition = nominal.step({"type": "submit"})
    final_info = _status(final_transition, "ok")
    _require(final_transition["terminated"] is True and final_transition["truncated"] is False,
             "valid nominal episode did not terminate normally")
    _require(_finite_score(final_transition.get("reward")), "nominal final reward is not finite")
    _require(final_info.get("exclude_from_rl") is True, "smoke final result is not excluded from RL")
    _require("smoke_profile" in final_info.get("rl_exclusion_reasons", []), "smoke exclusion reason is missing")
    _expect_post_final_error(nominal, workspace)
    checks["nominal_ordinary_seed23_test_and_final"] = "passed"
    checks["smoke_result_excluded_from_rl"] = "passed"
    checks["post_final_runtime_error"] = "passed"

    prior_artifacts = _artifact_hashes(workspace)
    reset_observation = nominal.reset(seed=23)
    _require(reset_observation["episode_index"] == 1, "reset did not create a fresh episode")
    _require(all(_artifact_hashes(workspace).get(path) == entry for path, entry in prior_artifacts.items()),
             "reset removed or changed prior episode artifacts")
    checks["reset_preserves_prior_artifacts"] = "passed"

    wide = environment_class(
        "softmax", workspace=workspace, seed=24, difficulty_id="wide/ordinary", **common
    )
    wide_observation = wide.reset(seed=24)
    _require(wide_observation["difficulty"]["id"] == "wide/ordinary", "wide ordinary case selection drifted")
    zero_candidate = "import torch\ndef kernel(x):\n    return torch.zeros_like(x)\n"
    wide_final = wide.step({"type": "submit", "code": zero_candidate})
    wide_info = _status(wide_final, "invalid", reward=0.0)
    _require(wide_final["terminated"] is True and wide_final["truncated"] is False,
             "invalid wide episode did not terminate normally")
    _require("softmax row normalization error" in wide_info.get("metrics", {}).get("error", ""),
             "zero probabilities were not rejected by softmax row normalization")
    _require(wide_info.get("exclude_from_rl") is True, "smoke invalid result is not excluded from RL")
    _expect_post_final_error(wide, workspace)
    checks["wide_ordinary_zero_output_rejected_for_row_mass"] = "passed"
    return checks


def _write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def run_reproduction(output: Path, workspace: Path | None = None, *, environment_class=None) -> dict:
    """Run CPU-only smoke checks and write a new receipt plus workspace records."""
    output, workspace = _validate_destinations(Path(output), workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "error",
        "scope": "local CPU smoke compatibility only",
        "execution": {
            "device": "cpu",
            "accelerator_requested": False,
            "evaluation_timeout_seconds": EVALUATION_TIMEOUT_SECONDS,
            "episode_budget_seconds": EPISODE_BUDGET_SECONDS,
        },
        "limitations": {
            "hardware_timing_validated": False,
            "outer_weights_trained": False,
            "external_adoption_established": False,
        },
        "workspace": os.path.relpath(workspace, output.parent),
        "checks": {},
    }
    workspace_created = False
    try:
        workspace.mkdir(parents=True, exist_ok=False)
        workspace_created = True
        receipt["runtime"] = _runtime_details()
        receipt["checks"] = _run_checks(workspace, environment_class)
        receipt["status"] = "ok"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()

    try:
        records_path = workspace / "full-records.json"
        if workspace_created and workspace.is_dir():
            _write_json_new(records_path, _full_records(workspace))
            receipt["full_records"] = {
                "path": records_path.relative_to(workspace).as_posix(),
                "sha256": _sha256(records_path),
            }
            receipt["workspace_artifacts"] = _artifact_hashes(workspace)
    except Exception as error:
        receipt["status"] = "error"
        receipt.setdefault("error", f"{type(error).__name__}: {error}")
        receipt.setdefault("traceback", traceback.format_exc())

    _write_json_new(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt path")
    parser.add_argument("--workspace", type=Path, help="new artifact directory (created when omitted)")
    args = parser.parse_args()
    try:
        receipt = run_reproduction(args.output, args.workspace)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
