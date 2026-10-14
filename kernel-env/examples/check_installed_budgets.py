#!/usr/bin/env python3
"""Install a local wheel into a temporary target and probe its RL budgets.

Run from any working directory. The receipt is one JSON object on stdout.
Only CPU work is used; the final adversarial kernel is stopped by its timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _wheel_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or path.suffix != ".whl":
            raise FileNotFoundError(f"not a wheel file: {path}")
        return path
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("rsi_kernel_env-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no rsi_kernel_env wheel under {root / 'dist'}")
    return max(wheels, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="wheel path (default: newest local build)")
    args = parser.parse_args()
    started = time.monotonic()
    receipt = {
        "status": "error",
        "checks": {},
        "python": sys.version.split()[0],
    }
    try:
        wheel = _wheel_path(args.wheel)
        receipt["wheel"] = str(wheel)
        with tempfile.TemporaryDirectory(prefix="kernel-installed-budget-") as scratch:
            scratch_path = Path(scratch)
            install_target = scratch_path / "installed"
            workdir = scratch_path / "cwd"
            install_target.mkdir()
            workdir.mkdir()
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--no-warn-script-location",
                    "--target",
                    str(install_target),
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            require(
                install.returncode == 0,
                f"pip install failed: {install.stderr[-2000:]}",
            )
            sys.path.insert(0, str(install_target))
            os.chdir(workdir)

            import kernel_env
            import kernel_env.environment as environment
            from kernel_env import KernelEnvironment

            package_path = Path(kernel_env.__file__).resolve()
            require(
                package_path.is_relative_to(install_target.resolve()),
                f"import resolved outside temporary wheel install: {package_path}",
            )
            receipt["imported_package"] = str(package_path)
            receipt["installed_version"] = importlib.metadata.version("rsi-kernel-env")
            receipt["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()

            original_evaluate = environment.evaluate
            mock_calls: list[dict] = []

            def fake_evaluate(solution, task, **kwargs):
                mock_calls.append({"task": task, **kwargs})
                return environment._record(
                    task,
                    kwargs["profile"],
                    "ok",
                    0.5,
                    {"mocked": True, "scored": True},
                    {"phase": kwargs["phase"], "seed": kwargs["seed"]},
                )

            environment.evaluate = fake_evaluate
            try:
                env = KernelEnvironment(
                    "softmax",
                    workspace=workdir / "deadline",
                    profile="standard",
                    split="train",
                    max_episode_seconds=5,
                )
                initial = env.reset()
                candidate = Path(initial["candidate"])
                starter_bytes = candidate.read_bytes()
                env.started = time.monotonic() - env.max_episode_seconds - 1
                expired = env.step({"type": "edit", "code": "def kernel(x): return x"})
                require(expired["info"]["status"] == "blocked", "expired action was not blocked")
                require(expired["reward"] is None and expired["truncated"], "expired reward was not null/truncated")
                require(candidate.read_bytes() == starter_bytes, "expired action overwrote source")
                require(not mock_calls, "expired action reached the evaluator")
                receipt["checks"]["deadline_before_write"] = "passed"

                malformed = KernelEnvironment(
                    "softmax",
                    workspace=workdir / "schema",
                    profile="standard",
                    split="train",
                    max_steps=2,
                )
                malformed.reset()
                first = malformed.step(None)
                last = malformed.step({"type": "edit"})
                require(first["info"]["status"] == "invalid", "bad action type was not invalid")
                require(first["reward"] == 0 and not first["terminated"], "mid-episode invalid reward/budget mismatch")
                require(last["info"]["status"] == "invalid", "missing edit code was not invalid")
                require(last["reward"] == 0 and last["terminated"], "last invalid action did not terminate at zero")
                require(last["info"]["provenance"]["stage"] == "policy_schema", "missing parser stage")
                require(last["info"]["metrics"]["scored"] is False, "parser record claims kernel was scored")
                require(not mock_calls, "malformed action reached the evaluator")
                with _expect_runtime_error():
                    malformed.step({"type": "submit"})
                receipt["checks"]["malformed_actions_consume_budget"] = "passed"

                seed_env = KernelEnvironment(
                    "softmax",
                    workspace=workdir / "seed",
                    profile="standard",
                    split="train",
                    seed=999_999_999,
                    max_steps=2,
                )
                seed_env.reset()
                seed_step = seed_env.step({"type": "test"})
                require(seed_step["info"]["status"] == "ok", "boundary seed step failed")
                require(0 <= mock_calls[-1]["seed"] < 1_000_000_000, "dev seed escaped supported range")
                receipt["checks"]["boundary_seed"] = "passed"
            finally:
                environment.evaluate = original_evaluate

            timeout_env = KernelEnvironment(
                "softmax",
                workspace=workdir / "timeout",
                profile="smoke",
                split="train",
                timeout=8,
                max_steps=1,
            )
            timeout_env.reset()
            infinite = (
                "def kernel(x):\n"
                "    print('ENTERED_KERNEL', flush=True)\n"
                "    while True:\n"
                "        pass\n"
            )
            timeout_started = time.monotonic()
            timed = timeout_env.step({"type": "submit", "code": infinite})
            timeout_elapsed = time.monotonic() - timeout_started
            require(timed["info"]["status"] == "error", "infinite candidate did not time out as error")
            require(timed["reward"] is None and timed["truncated"], "timeout reward was not null/truncated")
            require(timed["exclude_from_rl"] and timed["info"]["exclude_from_rl"], "timeout is RL-eligible")
            logs = list(timeout_env.episode_dir.glob("evidence/*/candidate.stdout.log"))
            entered = any("ENTERED_KERNEL" in path.read_text(errors="replace") for path in logs)
            require(entered, "candidate never entered kernel before timeout")
            receipt["checks"]["candidate_entered_kernel"] = "passed"
            receipt["candidate_timeout_elapsed_s"] = round(timeout_elapsed, 3)
            receipt["checks"]["candidate_timeout"] = "passed"

        receipt["status"] = "ok"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
    receipt["elapsed_s"] = round(time.monotonic() - started, 3)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "ok" else 1


class _expect_runtime_error:
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, _traceback):
        if error_type is None or not issubclass(error_type, RuntimeError):
            raise AssertionError("expected RuntimeError after episode completion")
        return True


if __name__ == "__main__":
    raise SystemExit(main())
