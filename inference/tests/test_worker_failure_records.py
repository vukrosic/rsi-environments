"""Validation and attribution tests for bounded worker failure receipts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorer
from assets import sha


def valid_receipt(source, pid, **overrides):
    value = {
        "schema_version": 1,
        "pid": pid,
        "source_sha256": sha(source),
        "stage": "prefill",
        "classification": "candidate",
        "error": {"type": "RuntimeError", "message": "candidate operation failed", "origin": "submission"},
    }
    value.update(overrides)
    return value


class WorkerFailureReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.source = root / "submission.py"
        self.source.write_text("class Engine: pass\n")
        self.output = root / "candidate.pt"
        self.pid = 12345
        self.receipt_path = self.output.with_suffix(".failure.json")

    def write(self, value):
        self.receipt_path.write_text(json.dumps(value))

    def assert_unknown(self, value, *, returncode=1):
        self.write(value)
        result = scorer.read_worker_failure(self.output, self.source, self.pid, returncode)
        self.assertEqual(result["classification"], "unknown")

    def test_missing_receipt_has_no_attribution(self):
        self.assertIsNone(scorer.read_worker_failure(self.output, self.source, self.pid, 1))

    def test_direct_submission_failures_are_candidate_attributed(self):
        for stage in ("candidate_import", "engine_init", "prefill", "decode"):
            with self.subTest(stage=stage):
                self.write(valid_receipt(self.source, self.pid, stage=stage))
                value = scorer.read_worker_failure(self.output, self.source, self.pid, 1)
                self.assertEqual(value["classification"], "candidate")
                self.assertEqual(value["stage"], stage)

    def test_candidate_contract_receipt_requires_contract_origin(self):
        self.write(valid_receipt(self.source, self.pid, stage="candidate_contract",
                                 error={"type": "ValueError", "message": "bad output", "origin": "contract"}))
        self.assertEqual(scorer.read_worker_failure(self.output, self.source, self.pid, 1)["classification"], "candidate")

    def test_candidate_attribution_rejects_unsupported_stage_origin_and_exit_status(self):
        cases = [
            (valid_receipt(self.source, self.pid, stage="model_load"), 1),
            (valid_receipt(self.source, self.pid, stage="prefill",
                           error={"type": "RuntimeError", "message": "worker failure", "origin": "worker"}), 1),
            (valid_receipt(self.source, self.pid, stage="candidate_contract"), 1),
            (valid_receipt(self.source, self.pid), 0),
            (valid_receipt(self.source, self.pid), -9),
        ]
        for value, returncode in cases:
            with self.subTest(stage=value["stage"], origin=value["error"]["origin"], returncode=returncode):
                self.assert_unknown(value, returncode=returncode)

    def test_large_receipt_is_unknown(self):
        self.receipt_path.write_text("{" + " " * 16385)
        result = scorer.read_worker_failure(self.output, self.source, self.pid, 1)
        self.assertEqual(result["classification"], "unknown")

    def test_source_hash_and_pid_are_bound_to_expected_worker(self):
        wrong_hash = valid_receipt(self.source, self.pid, source_sha256="0" * 64)
        wrong_pid = valid_receipt(self.source, self.pid + 1)
        self.assert_unknown(wrong_hash)
        self.assert_unknown(wrong_pid)

    def test_schema_types_stage_and_error_fields_are_validated(self):
        base = valid_receipt(self.source, self.pid)
        cases = []
        for key, value in (("schema_version", True), ("pid", True), ("source_sha256", []),
                           ("stage", 3), ("classification", 2),
                           ("stage", "made_up_stage"), ("error", "oops")):
            changed = copy.deepcopy(base)
            changed[key] = value
            cases.append((key, changed))
        for field, value in (("type", ""), ("type", "X" * 129),
                             ("message", "X" * 2049), ("origin", "candidate")):
            changed = copy.deepcopy(base)
            changed["error"][field] = value
            cases.append((f"error.{field}", changed))
        for label, value in cases:
            with self.subTest(field=label):
                self.assert_unknown(value)


if __name__ == "__main__":
    unittest.main()
