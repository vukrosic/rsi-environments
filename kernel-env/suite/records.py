"""Versioned score records; infrastructure failures are not training rewards.

MIT License. Copyright (c) 2026 Vuk Rosić.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
ENVIRONMENTS = {"inference", "training", "kernel"}
STATUSES = {"ok", "invalid", "blocked", "error"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict) -> dict:
    """Validate the envelope and reject NaN anywhere in the serialized record."""
    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")
    required = {"schema_version", "environment", "task", "profile", "status", "reward", "metrics", "provenance"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"missing record fields: {sorted(missing)}")
    if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if record["environment"] not in ENVIRONMENTS:
        raise ValueError("unknown environment")
    if not isinstance(record["task"], str) or not record["task"].strip():
        raise ValueError("task must be a nonempty string")
    if record["profile"] not in {"smoke", "standard"}:
        raise ValueError("profile must be smoke or standard")
    if record["status"] not in STATUSES:
        raise ValueError("unknown status")
    reward = record["reward"]
    if record["status"] in {"blocked", "error"}:
        if reward is not None:
            raise ValueError("blocked/error runs must have null reward")
    elif isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("ok/invalid runs need a finite reward in [0, 1]")
    if record["status"] == "invalid" and reward != 0:
        raise ValueError("invalid candidates must have reward 0")
    for field in ("metrics", "provenance"):
        if not isinstance(record[field], dict):
            raise ValueError(f"{field} must be a dictionary")
    # Return a detached JSON value so callers cannot mutate a stored observation.
    try:
        return json.loads(json.dumps(record, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"record must contain finite JSON values: {error}") from error


def make_record(*, environment: str, task: str, profile: str, status: str,
                reward: float | None, metrics: dict, provenance: dict) -> dict:
    return validate_record(dict(schema_version=SCHEMA_VERSION, environment=environment,
                                task=task, profile=profile, status=status, reward=reward,
                                metrics=metrics, provenance=provenance))


def speed_reward(speedup: float) -> float:
    if isinstance(speedup, bool) or not isinstance(speedup, (int, float)) or not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("speedup must be finite and positive")
    return 1 - 1 / (1 + speedup)


def loss_reward(loss: float) -> float:
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("held-out cross-entropy must be finite and nonnegative")
    return math.exp(-loss)


def write_record(path: str | Path, record: dict) -> None:
    """Atomically replace one validated JSON record in the destination directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(validate_record(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
