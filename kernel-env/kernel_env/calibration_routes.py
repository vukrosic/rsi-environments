"""Select a recorded calibration through one operator-supplied path.

An index only routes to an existing raw receipt. Timing, hardware equality and
source integrity remain the scorer's responsibility. Hashes and snapshots bind
its parsed input to the exact bytes read, without reopening mutable files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from . import difficulty

MAX_BYTES = 2 * 1024 * 1024
INDEX_KIND = "kernel_calibration_index"
_INDEX_KEYS = {"schema_version", "kind", "task", "profile", "receipts"}


def _path(value, *, label="calibration path") -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a nonempty path") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a nonempty path")
    return Path(raw)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON value: {value}")
    return number


def _read(path: Path) -> tuple[dict, bytes]:
    # A bounded single read also handles a file growing after it was opened.
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"calibration file exceeds {MAX_BYTES} bytes: {path}")
    try:
        document = json.loads(data, object_pairs_hook=_pairs,
                              parse_constant=_constant, parse_float=_float)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid calibration JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("calibration JSON must be an object")
    return document, data


def _is_index(document: dict) -> bool:
    # A malformed index must not silently become a raw receipt.
    return "kind" in document or "receipts" in document


def _ids(task: str) -> set[str]:
    if task not in difficulty.TRAIN_TASKS:
        raise ValueError("calibration indices support training tasks only")
    return {entry["id"] for entry in difficulty.catalog(task, "standard")} | {"legacy"}


def _validate_index(document: dict, task: str) -> dict:
    ids = _ids(task)
    if set(document) != _INDEX_KEYS:
        raise ValueError("calibration index must have exactly the schema-1 fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported calibration index schema_version")
    if document["kind"] != INDEX_KIND:
        raise ValueError("unknown calibration index kind")
    if document["task"] != task:
        raise ValueError("calibration index task differs from requested task")
    if document["profile"] != "standard":
        raise ValueError("calibration index profile must be standard")
    receipts = document["receipts"]
    if not isinstance(receipts, dict):
        raise ValueError("calibration index receipts must be an object")
    for case_id, value in receipts.items():
        if case_id not in ids:
            raise ValueError(f"unknown calibration difficulty ID: {case_id!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"receipt path for {case_id!r} must be a nonempty string")
    return receipts


def load(path, task: str, difficulty_id: str | None = None) -> dict:
    """Read a raw receipt or select one index entry, returning exact evidence.

    ``None`` selects the backward-compatible ``legacy`` entry in an index.
    Raw receipts deliberately defer identity and timing checks to the scorer,
    including the existing held-out-task legacy behavior. Files are bounded to
    2 MiB and strictly parsed even on the raw route. Indexed receipt paths are
    relative to the index directory unless absolute. Nested indices are rejected.
    """
    source = _path(path).resolve()
    document, data = _read(source)
    snapshots = {"calibration-receipt.json": data}
    result = {"receipt": document, "path": str(source),
              "sha256": hashlib.sha256(data).hexdigest(), "snapshots": snapshots}
    if not _is_index(document):
        return result
    receipts = _validate_index(document, task)
    selected = "legacy" if difficulty_id is None else difficulty_id
    if not isinstance(selected, str) or selected not in _ids(task):
        raise ValueError(f"unknown calibration difficulty ID: {selected!r}")
    if selected not in receipts:
        raise ValueError(f"calibration index has no receipt for difficulty {selected!r}")
    receipt_path = Path(receipts[selected])
    if not receipt_path.is_absolute():
        receipt_path = source.parent / receipt_path
    receipt_path = receipt_path.resolve()
    if receipt_path == source:
        raise ValueError("calibration index cycle: receipt points to its index")
    receipt, receipt_data = _read(receipt_path)
    if _is_index(receipt):
        raise ValueError("nested calibration indices are not supported")
    return {"receipt": receipt, "path": str(receipt_path),
            "sha256": hashlib.sha256(receipt_data).hexdigest(),
            "index_path": str(source), "index_sha256": hashlib.sha256(data).hexdigest(),
            "snapshots": {"calibration-receipt.json": receipt_data,
                          "calibration-index.json": data}}


def _validate_builder_receipt(receipt: dict, task: str, case_id: str) -> None:
    if _is_index(receipt):
        raise ValueError("nested calibration indices are not supported")
    if receipt.get("status") != "ok":
        raise ValueError("calibration must be a successful receipt")
    if receipt.get("task") != task:
        raise ValueError("calibration receipt task differs from index task")
    if receipt.get("profile", "standard") != "standard":
        raise ValueError("calibration receipt profile must be standard")
    expected = {"id": "legacy"} if case_id == "legacy" else difficulty.choose(
        task, "standard", 0, case_id)
    if receipt.get("difficulty") != expected:
        raise ValueError("calibration receipt difficulty differs from canonical index entry")
    system = receipt.get("system")
    if not isinstance(system, dict) or system.get("device") not in ("cpu", "mps", "cuda"):
        raise ValueError("calibration receipt needs a system/device marker")
    times = receipt.get("baseline_ms_by_variant")
    if not isinstance(times, dict) or set(times) != {str(v) for v in (0, 4, 5, 6, 10, 11)}:
        raise ValueError("calibration needs all six declared shape variants")
    for value in times.values():
        try:
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and value > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("calibration latencies must be finite positive numbers")


def write_index(output, task: str, receipts: dict[str, str | os.PathLike[str]]) -> Path:
    """Create an index of existing raw receipts, refusing any output overwrite.

    Supplied relative source paths resolve from the caller's working directory;
    the written index records absolute paths. Basic success/workload/system
    markers and six positive latency values are checked, but the scorer must
    still validate timing conditions and source integrity before awarding a
    reward. No calibration measurements are made.
    """
    ids = _ids(task)
    destination = _path(output, label="index output")
    # Keep the output leaf unresolved so dangling symlinks are also preserved.
    destination = Path(os.path.abspath(destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"calibration index output already exists: {destination}")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError("receipts must be a nonempty difficulty-to-path dictionary")
    paths = {}
    for case_id, source in receipts.items():
        if not isinstance(case_id, str) or case_id not in ids:
            raise ValueError(f"unknown calibration difficulty ID: {case_id!r}")
        source = _path(source, label=f"receipt path for {case_id!r}").resolve()
        receipt, _ = _read(source)
        _validate_builder_receipt(receipt, task, case_id)
        paths[case_id] = str(source)
    document = {"schema_version": 1, "kind": INDEX_KIND, "task": task,
                "profile": "standard", "receipts": paths}
    data = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(data) > MAX_BYTES:
        raise ValueError(f"calibration index exceeds {MAX_BYTES} bytes")
    with destination.open("xb") as stream:
        stream.write(data)
    return destination
