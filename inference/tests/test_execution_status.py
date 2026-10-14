"""Distinguish candidate execution failures from worker/infrastructure failures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorer


class ExecutionStatusTests(unittest.TestCase):
    def run_failure(self, failing_stage, failure):
        def worker(source, inputs, output, device, deadline):
            if output.stem == failing_stage:
                raise failure
            return {"conditions": []}

        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, RSI_INFERENCE_MEASUREMENT_LOCK=str(Path(root) / "measurement.lock")), \
             patch.object(scorer, "local_model"), \
             patch.object(scorer, "machine", return_value={"blockers": []}), \
             patch.object(scorer, "make_inputs", return_value={"task_id": "fixture-train", "decode_steps": 8, "cases": []}), \
             patch.object(scorer, "child", side_effect=worker):
            return scorer._evaluate_with_errors(scorer.HERE / "solution.py", Path(root) / "episode")

    def write_worker_script(self, root, receipt):
        worker_dir = Path(root) / "child"
        worker_dir.mkdir()
        (worker_dir / "worker.py").write_text(
            "import hashlib, json, os, sys\n"
            "from pathlib import Path\n"
            f"value = {receipt!r}\n"
            "value['pid'] = os.getpid()\n"
            "value['source_sha256'] = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()\n"
            "Path(sys.argv[3]).with_suffix('.failure.json').write_text(json.dumps(value))\n"
            "sys.exit(1)\n"
        )
        source = Path(root) / "candidate.py"
        source.write_text("class Engine: pass\n")
        inputs = Path(root) / "inputs.pt"
        inputs.write_bytes(b"synthetic input marker")
        return worker_dir, source, inputs

    @staticmethod
    def candidate_receipt():
        return {
            "schema_version": 1,
            "pid": 1,
            "source_sha256": "placeholder",
            "stage": "candidate_import",
            "classification": "candidate",
            "error": {"type": "SyntaxError", "message": "submission could not be imported", "origin": "submission"},
        }

    def test_direct_candidate_failure_is_invalid_zero(self):
        result = self.run_failure("candidate", scorer.CandidateFailure("bad Engine"))
        self.assertEqual(result["record"]["status"], "invalid")
        self.assertEqual(result["record"]["reward"], 0)

    def test_generic_candidate_runtime_failure_is_infrastructure_null(self):
        result = self.run_failure("candidate", RuntimeError("worker exited without candidate attribution"))
        self.assertEqual(result["record"]["status"], "error")
        self.assertIsNone(result["record"]["reward"])

    def test_candidate_failure_during_baseline_worker_is_infrastructure_null(self):
        result = self.run_failure("baseline_before", scorer.CandidateFailure("unexpected baseline candidate label"))
        self.assertEqual(result["record"]["status"], "error")
        self.assertIsNone(result["record"]["reward"])

    def test_real_child_valid_candidate_receipt_raises_candidate_failure(self):
        with tempfile.TemporaryDirectory() as root:
            worker_dir, source, inputs = self.write_worker_script(root, self.candidate_receipt())
            output = worker_dir / "candidate.pt"
            with self.assertRaises(scorer.CandidateFailure) as raised:
                scorer.child(source, inputs, output, "cpu", time.monotonic() + 10)
            self.assertIn("candidate_import", str(raised.exception))
            report = json.loads(output.with_suffix(".worker.json").read_text())
            self.assertEqual(report["failure_report"]["classification"], "candidate")
            self.assertEqual(report["returncode"], 1)

    def test_real_child_invalid_attribution_raises_infrastructure_runtime_error(self):
        receipt = self.candidate_receipt()
        receipt["stage"] = "model_load"
        with tempfile.TemporaryDirectory() as root:
            worker_dir, source, inputs = self.write_worker_script(root, receipt)
            output = worker_dir / "candidate.pt"
            with self.assertRaises(RuntimeError) as raised:
                scorer.child(source, inputs, output, "cpu", time.monotonic() + 10)
            self.assertNotIsInstance(raised.exception, scorer.CandidateFailure)
            report = json.loads(output.with_suffix(".worker.json").read_text())
            self.assertEqual(report["failure_report"]["classification"], "unknown")

    def test_worker_timeout_stays_infrastructure_and_cleans_child(self):
        with tempfile.TemporaryDirectory() as root:
            worker_dir = Path(root) / "child"
            worker_dir.mkdir()
            (worker_dir / "worker.py").write_text("import time\ntime.sleep(10)\n")
            source = Path(root) / "candidate.py"
            source.write_text("class Engine: pass\n")
            inputs = Path(root) / "inputs.pt"
            inputs.write_bytes(b"synthetic input marker")
            output = worker_dir / "candidate.pt"
            with patch.dict(scorer.LIMITS, {"worker_wall_s": 0.1}):
                with self.assertRaises(TimeoutError) as raised:
                    scorer.child(source, inputs, output, "cpu", time.monotonic() + 10)
            self.assertNotIsInstance(raised.exception, scorer.CandidateFailure)
            report = json.loads(output.with_suffix(".worker.json").read_text())
            self.assertIsNone(report["failure_report"])
            self.assertNotEqual(report["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
