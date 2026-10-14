"""Native index/receipt gates with explicit synthetic measurements, no GPU."""
import hashlib
import json
from pathlib import Path

import pytest

from kernel_env import difficulty, evaluate
from kernel_env import calibration_routes as routes
import kernel_env.environment as envmod

ROOT = Path(__file__).resolve().parents[1]


def fixture_receipt(task, family, monkeypatch):
    system = {"device": "cpu", "fixture": "calibration_gate_only"}
    monkeypatch.setattr(envmod, "_system", lambda _device: system)
    return {"status": "ok", "task": task, "system": system,
            "tasks_sha256": envmod._hash(envmod.tasks.__file__),
            "scorer_sha256": envmod._integrity(),
            "difficulty": difficulty.choose(task, "standard", 0, family),
            "baseline_ms_by_variant": {str(v): 1.0 for v in (0, 4, 5, 6, 10, 11)}}


def test_selected_index_receipt_is_validated_and_byte_snapshots_are_preserved(tmp_path, monkeypatch):
    family = "wide/mixed_scale"
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(fixture_receipt("softmax", family, monkeypatch)))
    index = routes.write_index(tmp_path / "index.json", "softmax", {family: raw})
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    details = {}
    descriptor = difficulty.choose("softmax", "standard", 22, family)
    receipt = envmod._calibration_receipt(index, "softmax", "cpu", descriptor,
                                         envmod._integrity(), details=details, evidence_dir=evidence)
    assert receipt["difficulty"] == descriptor
    assert details["path"] == str(raw.resolve()) and details["index_path"] == str(index)
    assert (evidence / "calibration-receipt.json").read_bytes() == raw.read_bytes()
    assert (evidence / "calibration-index.json").read_bytes() == index.read_bytes()
    assert details["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    raw.write_text('{"replaced":true}')
    assert hashlib.sha256((evidence / "calibration-receipt.json").read_bytes()).hexdigest() == details["sha256"]


def mock_rows(monkeypatch):
    monkeypatch.setattr(envmod, "_load", lambda: {"active_processes": [], "heavy_processes": [], "synthetic": True})
    monkeypatch.setattr(envmod, "_run_cases", lambda *a, **k: [
        {"variant": 6, "candidate_ms": 1.0, "baseline_ms": 1.0,
         "baseline_before_ms": 1.0, "baseline_after_ms": 1.0, "speedup": 1.0,
         "error": 0.0, "synthetic_fixture_not_a_measurement": True}])


@pytest.mark.parametrize("defect", ("missing_family", "wrong_receipt_family", "wrong_system", "wrong_scorer_hash"))
def test_bad_index_route_cannot_earn_training_reward(defect, tmp_path, monkeypatch):
    family = "odd/offset"
    receipt = fixture_receipt("softmax", family, monkeypatch)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(receipt))
    index = routes.write_index(tmp_path / "index.json", "softmax", {family: raw})
    if defect == "missing_family":
        family = "wide/ordinary"
    elif defect == "wrong_receipt_family":
        receipt["difficulty"] = difficulty.choose("softmax", "standard", 0, "nominal/ordinary")
    elif defect == "wrong_system":
        receipt["system"] = {"device": "cuda"}
    else:
        receipt["scorer_sha256"] = {"different": "fixture"}
    raw.write_text(json.dumps(receipt))
    mock_rows(monkeypatch)
    result = evaluate(ROOT / "starters" / "softmax.py", "softmax", profile="standard", phase="final",
                      difficulty_id=family, calibration=index, output_dir=tmp_path / "runs")
    assert result["status"] == "blocked" and result["reward"] is None
    assert result["trainable"] is False and result["exclude_from_rl"] is True


def test_index_cli_assembles_existing_receipts_without_benchmark(tmp_path, monkeypatch, capsys):
    from kernel_env.cli import main
    family = "nominal/ordinary"
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(fixture_receipt("softmax", family, monkeypatch)))
    output = tmp_path / "index.json"
    monkeypatch.setattr("sys.argv", ["kernel-env", "index-calibrations", "softmax", "--receipt", f"{family}={raw}", "--output", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["measurements_created"] is False
    assert routes.load(output, "softmax", family)["receipt"]["difficulty"]["id"] == family
