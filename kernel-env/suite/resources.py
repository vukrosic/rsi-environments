"""A host-wide cooperative accelerator lease, not an execution sandbox.

MIT License. Copyright (c) 2026 Vuk Rosić.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path


class AcceleratorBusy(TimeoutError):
    """Another participating process holds the accelerator lease."""


def lock_path() -> Path:
    override = os.environ.get("RESEARCH_ENVS_ACCELERATOR_LOCK")
    path = Path(override).expanduser() if override else Path.home() / ".cache/vuk-research-envs/accelerator.lock"
    if not path.is_absolute():
        raise ValueError("RESEARCH_ENVS_ACCELERATOR_LOCK must be an absolute path")
    return path


@contextlib.contextmanager
def accelerator_lock(*, device: str = "mps", timeout_seconds: float = 30,
                     path: str | Path | None = None):
    """Serialize accelerator work across all participating chats/processes.

    The same lock covers MPS/CUDA so different environment processes on the
    Mac cannot choose conflicting lock names. CPU work takes no lease.
    Kernel-level flock release survives process crashes; never unlink the file.
    """
    if device != "cpu" and device != "mps" and not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError("device must be cpu, mps, cuda, or cuda:N")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int)) or not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be finite and nonnegative")
    if device == "cpu":
        yield {"device": device, "accelerator_lease": False}
        return
    target = Path(path).expanduser() if path is not None else lock_path()
    if not target.is_absolute():
        raise ValueError("accelerator lock path must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with target.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AcceleratorBusy(f"accelerator busy: {target}")
                time.sleep(min(0.05, remaining))
        try:
            lease = {"device": device, "accelerator_lease": True, "pid": os.getpid(),
                     "lock_path": str(target), "acquired_unix": time.time()}
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(lease) + "\n")
            handle.flush()
            yield lease
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def known_competitors() -> list[dict]:
    """Report recognizable competing workloads without killing or altering them.

    This is a partial inventory, not proof of an idle device. Timers must still
    calibrate against their reference and record the actual measurement noise.
    """
    result = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True,
                            text=True, timeout=5, check=True)
    markers = ("/game/leagueoflegends", "remotion", "blender", "ffmpeg")
    found = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit() and any(marker in parts[1].lower() for marker in markers):
            found.append({"pid": int(parts[0]), "executable": parts[1]})
    return found
