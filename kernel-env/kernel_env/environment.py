"""Parent-owned scoring, evidence, calibration, and a tiny agent interaction API."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from . import tasks, difficulty
from .worker import fingerprint, pack_arrays, receive, send, unpack_arrays


ROOT = Path(__file__).resolve().parents[1]
STARTERS = Path(__file__).resolve().parent / "starters"
TRAIN_TASKS = ("rmsnorm", "softmax", "swiglu", "cross_entropy")
EVAL_TASKS = ("attention", "newton_schulz")
torch.set_num_threads(1)


class InvalidCandidate(Exception):
    pass


class Blocked(Exception):
    pass


def _seed(value):
    if type(value) is not int or not 0 <= value < 1_000_000_000:
        raise ValueError("seed must be an integer in [0,1000000000)")
    return value


def _difficulty(task, profile, seed, difficulty_id):
    if difficulty_id is None:
        return None  # Raw scorer's backward-compatible workload.
    if task not in TRAIN_TASKS:
        raise ValueError("difficulty sampling is available only for training operations")
    return difficulty.choose(task, profile, seed, difficulty_id=difficulty_id)


def _inputs(task, profile, seed, variant, descriptor):
    if descriptor is None:
        xs = tasks.inputs(task, profile, seed, variant=variant)
        case = {"id": "legacy", "variant": variant,
                "shapes": [list(x.shape) for x in xs]}
    else:
        xs = difficulty.inputs(descriptor, seed, variant=variant)
        case = difficulty.case(descriptor, variant=variant)
    return xs, dict(case, seed=seed)


def _suite():
    # In-checkout use shares the coordinating chat's implementation. The
    # standalone release also packages a verbatim licensed copy under suite/.
    if not (ROOT / "suite" / "records.py").exists():
        sys.path.insert(0, str(ROOT.parent))
    from suite.records import make_record
    from suite.resources import accelerator_lock
    return make_record, accelerator_lock


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integrity():
    files = sorted((ROOT / "kernel_env").glob("*.py"))
    for folder in (ROOT / "suite", ROOT.parent / "suite"):
        if (folder / "records.py").exists():
            files += sorted(folder.glob("*.py"))
            break
    return {str(p): _hash(p) for p in files}


def _system(device):
    chip = platform.processor()
    if sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=3).stdout.strip()
    return {"device": device, "machine": platform.machine(), "chip": chip,
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "cpu_threads": 1,
            "timer": "parent wall clock: run command through synchronized done; IPC included",
            "dtype": "float32", "cuda_tf32": False}


def _load():
    """Cheap load evidence; cooperative locks cannot detect every GPU client."""
    rows = subprocess.run(["ps", "-axo", "pid,pcpu,command"], capture_output=True,
                          text=True, timeout=3).stdout.splitlines()[1:]
    heavy, active = [], []
    for row in rows:
        parts = row.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid, cpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid == os.getpid():
            continue
        if cpu > 50:
            active.append({"pid": pid, "cpu_percent": cpu, "program": Path(command.split()[0]).name})
        if (any(x in command.lower() for x in ("leagueoflegends", "chrome-headless-shell", "blender"))
                or ("ffmpeg" in command.lower() and cpu > 20)
                or ("python" in command and ("kernels/bench.py" in command or "kloop.py tick" in command))):
            heavy.append({"pid": pid, "program": Path(command.split()[0]).name, "cpu_percent": cpu})
    return {"load_average": list(os.getloadavg()), "cpu_count": os.cpu_count(),
            "heavy_processes": heavy, "active_processes": active,
            "idle_check_limit": "known render/game processes or unrelated process >50% CPU"}


def _idle_reason(load, excluded=()):
    contenders = [r for r in load["active_processes"] if r["pid"] not in excluded]
    if load["heavy_processes"]:
        return "known heavy application is running"
    if contenders:
        return "unrelated process exceeds 50% CPU"
    return None


def _check_device(device):
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise Blocked("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise Blocked("CUDA unavailable (CUDA path is unverified on this Mac)")


class _Runner:
    def __init__(self, mode, task, source, device, folder, deadline):
        self.deadline = deadline
        self.sock, child = socket.socketpair()
        self.stdout = open(folder / f"{mode}.stdout.log", "wb")
        self.stderr = open(folder / f"{mode}.stderr.log", "wb")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")
        env["PYTHONPATH"] = str(ROOT)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "kernel_env.worker", str(child.fileno()),
             device, mode, task, str(source)], cwd=folder, env=env,
            pass_fds=(child.fileno(),), stdout=self.stdout, stderr=self.stderr,
            start_new_session=True)
        child.close()
        try:
            if self.recv(16) != b"hello":
                raise ValueError("bad worker handshake")
        except BaseException:
            self.close()
            raise

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("evaluation wall-clock limit reached")
        self.sock.settimeout(left)

    def recv(self, maximum):
        self.remaining()
        return receive(self.sock, maximum, deadline=self.deadline)

    def call(self, arrays):
        self.remaining()
        send(self.sock, pack_arrays(arrays))
        if self.recv(16) != b"ready":
            raise ValueError("bad ready handshake")
        self.remaining()
        start = time.perf_counter_ns()
        send(self.sock, b"run")
        if self.recv(16) != b"done":
            raise ValueError("bad done handshake")
        ms = (time.perf_counter_ns() - start) / 1e6
        metadata = json.loads(self.recv(4096))
        if "error" in metadata:
            raise InvalidCandidate(metadata["error"])
        if metadata.get("dtype") != "torch.float32":
            raise InvalidCandidate("wrong output dtype; expected float32")
        # Outputs for these six tasks never exceed the largest input.
        cap = max(x.nbytes for x in arrays) + 65536
        try:
            values = unpack_arrays(self.recv(cap))
        except ValueError as exc:
            raise InvalidCandidate(f"invalid output protocol: {exc}") from exc
        if len(values) != 1:
            raise InvalidCandidate("expected one output array")
        return values[0], ms, metadata

    def close(self):
        # Terminate only the process group this evaluator created.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
        # Descendants can outlive an already-exited direct worker.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)
        self.sock.close()
        self.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _validate(actual, expected, metadata, arrays, device, tolerance, *, task=None):
    if metadata.get("input_hashes") != [fingerprint(x) for x in arrays]:
        raise InvalidCandidate("kernel mutated input tensors")
    if metadata.get("device") != device:
        raise InvalidCandidate("output must stay on the requested device")
    if actual.shape != expected.shape:
        raise InvalidCandidate(f"wrong output shape {actual.shape}; expected {expected.shape}")
    if actual.dtype != np.float32:
        raise InvalidCandidate(f"wrong output dtype {actual.dtype}; expected float32")
    if not np.isfinite(actual).all():
        raise InvalidCandidate("non-finite output")
    if task == "softmax":
        # Elementwise absolute tolerance alone allows whole low-probability
        # rows to disappear. Probability mass is a separate semantic invariant.
        if np.any(actual < 0) or np.any(actual > 1):
            raise InvalidCandidate("softmax output must contain probabilities in [0,1]")
        mass_error = float(np.max(np.abs(actual.sum(axis=-1, dtype=np.float64) - 1.0)))
        if mass_error > tolerance:
            raise InvalidCandidate(f"softmax row normalization error {mass_error:.7g} exceeds {tolerance}")
    error = float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))
    if error > tolerance:
        raise InvalidCandidate(f"correctness error {error:.7g} exceeds {tolerance}")
    return error


def _validate_baseline(actual, expected, metadata, arrays, device, tolerance, *, task=None):
    try:
        return _validate(actual, expected, metadata, arrays, device, tolerance, task=task)
    except InvalidCandidate as exc:
        raise Blocked(f"trusted PyTorch baseline failed correctness: {exc}") from exc


def _baseline_call(runner, arrays):
    # Protocol/output faults in the trusted comparator must never become a
    # zero-reward candidate failure before that candidate is even called.
    try:
        return runner.call(arrays)
    except InvalidCandidate as exc:
        raise Blocked(f"trusted PyTorch baseline failed output protocol: {exc}") from exc


def _run_cases(task, profile, phase, seed, trials, device, folder, snapshot, deadline,
               calibration_only=False, variants=None, descriptor=None):
    rows = []
    # Never time the all-zero stress fixtures: every scored input must carry
    # fresh random values. Final also uses different dimensions from dev.
    variants = variants or ((0, 4, 5) if phase == "dev" else (6, 10, 11))
    with contextlib.ExitStack() as stack:
        baseline = stack.enter_context(_Runner("baseline", task, snapshot, device, folder, deadline))
        candidate = None if calibration_only else stack.enter_context(
            _Runner("candidate", task, snapshot, device, folder, deadline))
        # Fresh seed ranges; final never overlaps development for the same seed.
        offset = 0 if phase == "dev" else 1_000_000_000
        # Unscored boundary cases test numerical correctness. All scored
        # cases below retain fresh random values; zero fixtures earn no speed.
        if candidate is not None:
            for variant in (1, 2, 3):
                case_seed = seed + offset + 5000 + variant
                xs, case = _inputs(task, profile, case_seed, variant, descriptor)
                arrays = tuple(x.numpy() for x in xs)
                (folder / "current_case.json").write_text(json.dumps({
                    "seed": case_seed, "variant": variant, "stage": "unscored_stress", "case": case}))
                with torch.no_grad():
                    expected = tasks.reference(task, *xs).numpy()
                out, _, metadata = _baseline_call(baseline, arrays)
                _validate_baseline(out, expected, metadata, arrays, device, tasks.tolerance(task), task=task)
                actual, _, metadata = candidate.call(arrays)
                _validate(actual, expected, metadata, arrays, device, tasks.tolerance(task), task=task)
        for i in range(trials + 2):
            variant = variants[0] if i < 2 else variants[(i - 2) % len(variants)]
            case_seed = seed + offset + 10_000 + i
            xs, case = _inputs(task, profile, case_seed, variant, descriptor)
            arrays = tuple(x.numpy() for x in xs)
            (folder / "current_case.json").write_text(json.dumps({
                "seed": case_seed, "variant": variant,
                "stage": "warmup" if i < 2 else "measured",
                "shapes": [list(x.shape) for x in arrays], "case": case}))
            with torch.no_grad():
                expected = tasks.reference(task, *xs).numpy()
            b_out, before, bm = _baseline_call(baseline, arrays)
            _validate_baseline(b_out, expected, bm, arrays, device, tasks.tolerance(task), task=task)
            row = {"seed": case_seed, "variant": variant,
                   "shapes": [list(x.shape) for x in arrays], "case": case,
                   "baseline_before_ms": before}
            if candidate is not None:
                actual, c_ms, cm = candidate.call(arrays)
                row["candidate_ms"] = c_ms
                row["error"] = _validate(actual, expected, cm, arrays, device, tasks.tolerance(task), task=task)
                # The candidate never sees this second baseline call; fresh
                # values avoid a baseline/result cache and bracket drift.
                other, after_case = _inputs(task, profile, case_seed + 100_000_000, variant, descriptor)
                other_arrays = tuple(x.numpy() for x in other)
                with torch.no_grad():
                    other_expected = tasks.reference(task, *other).numpy()
                after_out, after, am = _baseline_call(baseline, other_arrays)
                _validate_baseline(after_out, other_expected, am, other_arrays, device, tasks.tolerance(task), task=task)
                row["baseline_after_ms"] = after
                row["baseline_after_case"] = after_case
                row["baseline_ms"] = (before + after) / 2
                row["speedup"] = row["baseline_ms"] / c_ms
            if i >= 2:
                rows.append(row)
            # Durable partial evidence survives an invalid later call.
            (folder / "trials.json").write_text(json.dumps(rows, indent=2))
        return rows


def _rl_metadata(task, profile, status, *, terminal):
    """Conservative public mask, independent of a caller's requested split.

    An evaluation score may remain numeric for reporting. It is never an RL
    update target. Smoke and intermediate feedback are also excluded.
    """
    split = "train" if task in TRAIN_TASKS else "eval"
    reasons = []
    if split == "eval":
        reasons.append("evaluation_only_task")
    if status in ("blocked", "error"):
        reasons.append("infrastructure_or_execution_failure")
    elif status not in ("ok", "invalid"):
        reasons.append("no_measured_score")
    if not terminal:
        reasons.append("nonterminal_feedback")
    if profile != "standard":
        reasons.append("smoke_profile")
    return {"task_split": split, "trainable": not reasons,
            "exclude_from_rl": bool(reasons), "rl_exclusion_reasons": reasons}


def _record(task, profile, status, reward, metrics, provenance):
    make_record, _ = _suite()
    metadata = _rl_metadata(task, profile, status, terminal=provenance.get("phase") == "final")
    provenance = dict(provenance, partition=metadata["task_split"])
    record = make_record(environment="kernel", task=task, profile=profile,
                         status=status, reward=reward, metrics=metrics, provenance=provenance)
    return dict(record, **metadata)


def _calibration_receipt(path, task, device, descriptor, integrity):
    """Reject malformed/nonfinite calibration rather than bypassing drift gates."""
    try:
        def constant(value):
            raise ValueError(f"nonfinite JSON constant {value}")
        receipt = json.loads(Path(path).read_text(), parse_constant=constant)
        if not isinstance(receipt, dict) or receipt.get("status") != "ok":
            raise ValueError("calibration must be a successful receipt")
        if receipt.get("system") != _system(device):
            raise ValueError("calibration hardware/software/device differs")
        if receipt.get("task") != task or receipt.get("tasks_sha256") != _hash(tasks.__file__):
            raise ValueError("calibration task or baseline source differs")
        if receipt.get("scorer_sha256") != integrity:
            raise ValueError("calibration timer/worker/scorer source differs")
        if receipt.get("difficulty") != (descriptor or {"id": "legacy"}):
            raise ValueError("calibration difficulty differs from evaluated workload")
        times = receipt.get("baseline_ms_by_variant")
        required = {str(v) for v in (0, 4, 5, 6, 10, 11)}
        if not isinstance(times, dict) or set(times) != required:
            raise ValueError("calibration needs all six declared shape variants")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
               for x in times.values()):
            raise ValueError("calibration latencies must be finite positive numbers")
        return receipt
    except (OSError, ValueError, TypeError) as exc:
        raise Blocked(f"invalid calibration receipt: {exc}") from exc


def _config(task, profile, phase, trials, timeout):
    if task not in tasks.NAMES:
        raise ValueError(f"unknown task {task}")
    if profile not in ("smoke", "standard") or phase not in ("dev", "final"):
        raise ValueError("profile smoke/standard and phase dev/final required")
    if trials is None:
        trials = 3 if phase == "dev" else 5
    if type(trials) is not int or trials < 1 or trials > 30:
        raise ValueError("trials must be an integer in [1,30]")
    if profile == "standard" and phase == "final" and trials != 5:
        raise ValueError("standard final uses exactly five measured trials")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be in (0,3600]")
    return trials


def evaluate(solution, task, *, device="cpu", profile="smoke", phase="dev", seed=0,
             trials=None, timeout=120, output_dir=None, calibration=None, lock_timeout=2,
             difficulty_id=None):
    """Freeze and score one single-file candidate. Return a suite v1 record.

    Invalid outputs receive zero reward. Runtime failures/timeouts and timing
    infrastructure blocks receive null. Every invocation retains evidence.
    """
    trials = _config(task, profile, phase, trials, timeout)
    seed = _seed(seed)
    descriptor = _difficulty(task, profile, seed, difficulty_id)
    started = time.monotonic()
    deadline = started + timeout
    run_id = uuid.uuid4().hex
    folder = (Path(output_dir or Path.cwd() / "runs") / run_id).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    source = Path(solution).read_bytes()
    if len(source) > 2 * 1024 * 1024:
        raise ValueError("candidate exceeds 2 MiB single-file limit")
    digest = hashlib.sha256(source).hexdigest()
    snapshot = folder / f"candidate-{digest}.py"
    snapshot.write_bytes(source)
    snapshot.chmod(0o444)
    integrity = _integrity()
    provenance = dict(_system(device), source_sha256=digest, seed=seed, phase=phase,
                      run_id=run_id, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      source_snapshot=str(snapshot.resolve()), scorer_sha256=integrity,
                      trials=trials, timeout_s=timeout, difficulty=descriptor or {"id": "legacy"})
    metrics = {"baseline": tasks.baseline_name(task), "tolerance": tasks.tolerance(task),
               "error_definition": "max(abs(actual-expected)/max(1,abs(expected)))",
               "correctness_passed": False, "timing_reliable": False,
               "development_only": profile == "smoke"}
    (folder / "config.json").write_text(json.dumps(provenance, indent=2))
    status, reward = "error", None
    rows = []
    try:
        _check_device(device)
        _, accelerator_lock = _suite()
        context = accelerator_lock(device=device, timeout_seconds=min(lock_timeout, max(0, deadline - time.monotonic()))) if device != "cpu" else contextlib.nullcontext()
        with context:
            metrics["load_before"] = _load()
            if device != "cpu" and metrics["load_before"]["heavy_processes"]:
                raise Blocked("recognized accelerator workload is running; GPU test deferred")
            rows = _run_cases(task, profile, phase, seed, trials, device, folder, snapshot, deadline,
                              descriptor=descriptor)
            metrics["load_after"] = _load()
        metrics.update(correctness_passed=True, trials=rows,
                       candidate_ms=statistics.median(r["candidate_ms"] for r in rows),
                       baseline_ms=statistics.median(r["baseline_ms"] for r in rows),
                       speedup=statistics.median(r["speedup"] for r in rows),
                       worst_rel_error=max(r["error"] for r in rows))
        if _hash(snapshot) != digest:
            raise InvalidCandidate("candidate snapshot changed during evaluation")
        if _integrity() != integrity:
            raise Blocked("trusted scorer source changed during evaluation")
        if profile == "standard":
            reason = _idle_reason(metrics["load_before"]) or _idle_reason(metrics["load_after"])
            if reason:
                raise Blocked(reason)
            if calibration is None:
                raise Blocked("standard timing requires a fixed idle calibration receipt")
            receipt_path = Path(calibration)
            receipt = _calibration_receipt(receipt_path, task, device, descriptor, integrity)
            metrics["calibration"] = {"path": str(receipt_path.resolve()), "sha256": _hash(receipt_path)}
            for row in rows:
                idle = receipt["baseline_ms_by_variant"][str(row["variant"])]
                drift = max(row["baseline_before_ms"], row["baseline_after_ms"]) / idle
                pair = row["baseline_before_ms"] / row["baseline_after_ms"]
                if drift > 1.10 or not 0.90 <= pair <= 1.10:
                    raise Blocked("baseline drift/noise exceeds predeclared 10% limit")
            metrics["timing_reliable"] = True
        status = "ok"
        reward = metrics["speedup"] / (1 + metrics["speedup"])
    except InvalidCandidate as exc:
        status, reward = "invalid", 0.0
        metrics["error"] = str(exc)
    except (Blocked, TimeoutError) as exc:
        # Lock timeout is infrastructure; candidate wall timeout is execution.
        status = "blocked" if isinstance(exc, Blocked) or type(exc).__name__ == "AcceleratorBusy" else "error"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    metrics["elapsed_s"] = time.monotonic() - started
    if not rows and (folder / "trials.json").exists():
        metrics["partial_trials"] = json.loads((folder / "trials.json").read_text())
    if (folder / "current_case.json").exists():
        metrics["last_case"] = json.loads((folder / "current_case.json").read_text())
    result = _record(task, profile, status, reward, metrics, provenance)
    (folder / "record.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def calibrate(task, *, device="cpu", seed=2301, output_dir=None, timeout=180,
              lock_timeout=2, difficulty_id=None):
    """Measure a baseline on an inspected idle machine; never replace old receipts."""
    _config(task, "standard", "dev", 18, timeout)
    seed = _seed(seed)
    descriptor = _difficulty(task, "standard", seed, difficulty_id)
    folder = (Path(output_dir or Path.cwd() / "runs") / ("calibration-" + uuid.uuid4().hex)).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "blocked", "task": task, "system": _system(device),
               "tasks_sha256": _hash(tasks.__file__), "scorer_sha256": _integrity(), "seed": seed,
               "difficulty": descriptor or {"id": "legacy"},
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    deadline = time.monotonic() + timeout
    try:
        _check_device(device)
        _, accelerator_lock = _suite()
        ctx = accelerator_lock(device=device, timeout_seconds=min(lock_timeout, max(0, deadline - time.monotonic()))) if device != "cpu" else contextlib.nullcontext()
        with ctx:
            receipt["load_before"] = _load()
            reason = _idle_reason(receipt["load_before"])
            if reason:
                raise Blocked(reason)
            rows = _run_cases(task, "standard", "dev", seed, 18, device, folder, Path("unused"),
                              deadline, calibration_only=True,
                              variants=(0, 4, 5, 6, 10, 11), descriptor=descriptor)
            receipt["load_after"] = _load()
        reason = _idle_reason(receipt["load_after"])
        if reason:
            raise Blocked(reason)
        receipt["trials"] = rows
        medians = {}
        for variant in (0, 4, 5, 6, 10, 11):
            times = [r["baseline_before_ms"] for r in rows if r["variant"] == variant]
            medians[str(variant)] = statistics.median(times)
            if max(times) / min(times) > 1.10:
                raise Blocked("idle baseline spread exceeds 10%; calibration not reliable")
        receipt.update(status="ok", baseline_ms_by_variant=medians)
        if receipt["scorer_sha256"] != _integrity():
            raise Blocked("scorer source changed during calibration")
    except Exception as exc:
        receipt["status"] = "blocked"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    path = folder / "calibration.json"
    receipt["path"] = str(path.resolve())
    path.write_text(json.dumps(receipt, indent=2))
    return receipt


class KernelEnvironment:
    """Budgeted environment-side RL API; it does not update model weights.

    reset() -> observation dict; step(action) -> transition dict containing
    observation, reward, terminated, truncated, info. Rewards are zero before
    submission; terminal reward is measured on final cases. Infrastructure
    failures truncate with null reward and must be excluded by the trainer.
    """
    def __init__(self, task, workspace, *, device="cpu", profile="smoke", seed=0,
                 calibration=None, timeout=120, split=None, max_steps=5,
                 max_episode_seconds=600, difficulty_id=None):
        _config(task, profile, "dev", None, timeout)
        seed = _seed(seed)
        if split is None:
            split = "train" if task in TRAIN_TASKS else "eval"
        if split not in ("train", "eval") or task not in (TRAIN_TASKS if split == "train" else EVAL_TASKS):
            raise ValueError("task does not belong to requested train/eval split")
        if type(max_steps) is not int or not 1 <= max_steps <= 1000:
            raise ValueError("max_steps must be an integer in [1,1000]")
        if isinstance(max_episode_seconds, bool) or not isinstance(max_episode_seconds, (int, float)) or not math.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        if difficulty_id is not None:
            _difficulty(task, profile, seed, difficulty_id)
        self.task, self.workspace = task, Path(workspace)
        self.config = dict(device=device, profile=profile, seed=seed,
                           calibration=calibration, timeout=timeout)
        self.split, self.max_steps = split, max_steps
        self.max_episode_seconds = max_episode_seconds
        self.difficulty_id = difficulty_id
        self.episode_index = -1
        self.steps = 0
        self.done = True

    def reset(self, *, seed=None):
        self.episode_index += 1
        self.active_seed = (_seed(seed) if seed is not None else
                            (self.config["seed"] + self.episode_index) % 1_000_000_000
                            if self.task in TRAIN_TASKS else self.config["seed"])
        self.difficulty = (difficulty.choose(self.task, self.config["profile"], self.active_seed,
                                             difficulty_id=self.difficulty_id)
                           if self.task in TRAIN_TASKS else None)
        self.episode_dir = self.workspace / "episodes" / uuid.uuid4().hex
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.candidate = self.episode_dir / "candidate.py"
        self.candidate.write_text((STARTERS / f"{self.task}.py").read_text())
        self.steps, self.done, self.started = 0, False, time.monotonic()
        return self._observation()

    def _observation(self, feedback=None):
        return {"task": self.task, "split": self.split,
                "task_spec": tasks.describe(self.task, self.config["profile"]),
                "seed": self.active_seed, "episode_index": self.episode_index,
                "difficulty": self.difficulty or {"id": "legacy", "sampling": "frozen_eval"},
                "candidate": str(self.candidate.resolve()), "code": self.candidate.read_text(),
                "profile": self.config["profile"], "device": self.config["device"],
                "tolerance": tasks.tolerance(self.task),
                "remaining_steps": max(0, self.max_steps - self.steps),
                "remaining_seconds": max(0, self.max_episode_seconds - (time.monotonic() - self.started)),
                "feedback": feedback,
                "instruction": "kernel(*inputs) must preserve float32 inputs and return a correct same-device tensor. Actions: edit(code), test(optional code), submit(optional code). Last budgeted action submits automatically. Final reward uses fresh held-out shapes/seeds."}

    def _failure_record(self, status, reward, error, *, stage, terminal):
        return _record(self.task, self.config["profile"], status, reward,
                       {"error": error, "scored": False},
                       {"phase": "final" if terminal else "dev", "seed": self.active_seed,
                        "stage": stage, "difficulty": self.difficulty or {"id": "legacy"}})

    def _transition(self, info, *, reward, terminated, truncated):
        # Repeat the mask at both public boundaries. Generic trainers often
        # store transitions while offline consumers store only score records.
        metadata = _rl_metadata(self.task, self.config["profile"], info["status"],
                                terminal=terminated and not truncated)
        info = dict(info, **metadata)
        transition = {"observation": self._observation(info), "reward": reward,
                      "terminated": terminated, "truncated": truncated,
                      "info": info, **metadata}
        (self.episode_dir / f"transition-{self.steps:02d}.json").write_text(
            json.dumps(transition, indent=2, allow_nan=False))
        return transition

    def step(self, action):
        if self.done:
            raise RuntimeError("episode ended or was not reset; call reset()")
        # Count every attempted action, including malformed policy output.
        # Check the wall deadline before validation or filesystem writes.
        self.steps += 1
        remaining = self.max_episode_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            self.done = True
            record = self._failure_record("blocked", None, "episode wall budget exhausted",
                                          stage="episode_budget", terminal=True)
            return self._transition(record, reward=None, terminated=False, truncated=True)
        if isinstance(action, str):
            action = {"type": "test", "code": action}
        try:
            if (not isinstance(action, dict) or set(action) - {"type", "code"}
                    or action.get("type") not in ("edit", "test", "submit")):
                raise ValueError("action requires type edit/test/submit and optional code string, without extra keys")
            code = action.get("code")
            if action["type"] == "edit" and code is None:
                raise ValueError("edit requires code")
            if code is not None:
                if not isinstance(code, str) or not code.strip() or len(code.encode()) > 2 * 1024 * 1024:
                    raise ValueError("code must be nonempty UTF-8 source at most 2 MiB")
        except (ValueError, TypeError, UnicodeError) as exc:
            terminal = self.steps >= self.max_steps
            self.done = terminal
            record = self._failure_record("invalid", 0.0, str(exc), stage="policy_schema", terminal=terminal)
            return self._transition(record, reward=0.0, terminated=terminal, truncated=False)
        if code is not None:
            try:
                self.candidate.write_text(code)
            except OSError as exc:
                self.done = True
                record = self._failure_record("error", None, str(exc), stage="candidate_write", terminal=True)
                return self._transition(record, reward=None, terminated=False, truncated=True)
        terminal = action["type"] == "submit" or self.steps >= self.max_steps
        if not terminal and action["type"] == "edit":
            return self._transition({"status": "edited"}, reward=0.0,
                                    terminated=False, truncated=False)
        remaining = self.max_episode_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            self.done = True
            record = self._failure_record("blocked", None, "episode wall budget exhausted during action",
                                          stage="episode_budget", terminal=True)
            return self._transition(record, reward=None, terminated=False, truncated=True)
        config = dict(self.config, timeout=min(self.config["timeout"], remaining), seed=self.active_seed)
        if self.difficulty is not None:
            config["difficulty_id"] = self.difficulty["id"]
        if not terminal:
            config["seed"] = (self.active_seed + self.steps * 1000) % 1_000_000_000
        record = evaluate(self.candidate, self.task, phase="final" if terminal else "dev",
                          output_dir=self.episode_dir / "evidence", **config)
        interrupted = record["status"] in ("blocked", "error")
        self.done = terminal or interrupted
        return self._transition(record,
                                reward=None if interrupted else record["reward"] if terminal else 0.0,
                                terminated=terminal and not interrupted, truncated=interrupted)

    def final(self):
        return self.step({"type": "submit"})
