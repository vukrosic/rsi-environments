"""Saved-output replay keeps legacy V1 compatibility and validates V2 coverage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets import sha
from replay import replay
from test_artifact_integrity import make_artifact, task_fixture
from test_oracle_exclusion import artifact


class ReplayTests(unittest.TestCase):
    def fixture(self, root, *, version=1, oracle_offset=0, candidate_offset=0):
        (root / "submission.py").write_text("class Engine: pass\n")
        if version == 2:
            task = task_fixture()
            torch.save(task, root / "inputs.pt")
            artifacts = {
                "baseline_before": make_artifact(task),
                "candidate": make_artifact(task),
                "baseline_after": make_artifact(task),
            }
            if candidate_offset:
                artifacts["candidate"]["rows"][0]["first"][0, 1] += candidate_offset
            if oracle_offset:
                artifacts["baseline_after"]["rows"][0]["first"][0, 1] += oracle_offset
            task_id = "rsi.inference.smollm2-135m.v2"
        else:
            task = None
            torch.save({"cases": []}, root / "inputs.pt")
            artifacts = {
                "baseline_before": artifact(0.1),
                "candidate": artifact(0.1, candidate_offset),
                "baseline_after": artifact(0.1, oracle_offset),
            }
            task_id = "rsi.inference.smollm2-135m.v1"

        workers = {}
        for name, value in artifacts.items():
            torch.save(value, root / f"{name}.pt")
            workers[name] = {"artifact_sha256": sha(root / f"{name}.pt")}
        receipt = {
            "task": task_id,
            "status": "diagnostic_pass",
            "source_sha256": sha(root / "submission.py"),
            "evaluator_hashes": {"submission.py": sha(root / "submission.py")},
            "inputs_sha256": sha(root / "inputs.pt"),
            "workers": workers,
        }
        (root / "result.json").write_text(json.dumps(receipt))

    def test_v1_saved_run_remains_replayable_without_v2_task_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, version=1)
            result = replay(root)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["artifact_hashes_verified"])
            self.assertTrue(result["ok"])
            self.assertIsNone(result["reward"])
            self.assertFalse(result["model_executed"])
            self.assertNotIn("input_coverage_verified", result)

    def test_v2_replay_validates_saved_input_and_probe_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, version=2)
            result = replay(root)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["input_coverage_verified"])
            self.assertTrue(result["artifact_hashes_verified"])
            self.assertTrue(result["ok"])
            self.assertIsNone(result["reward"])
            self.assertFalse(result["model_executed"])

    def test_v1_and_v2_oracle_disagreement_remain_errors_with_null_reward(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, version=version, oracle_offset=1)
                result = replay(root)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["reward"])
                self.assertFalse(result["oracle_consistent"])

    def test_v1_and_v2_wrong_candidates_remain_invalid_without_new_reward(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, version=version, candidate_offset=1)
                result = replay(root)
                self.assertEqual(result["status"], "invalid")
                self.assertTrue(result["oracle_consistent"])
                self.assertIsNone(result["reward"])

    def test_modified_source_or_saved_tensor_is_refused(self):
        for name in ("submission.py", "candidate.pt", "inputs.pt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, version=2)
                with (root / name).open("ab") as output:
                    output.write(b"changed")
                result = replay(root)
                self.assertEqual(result["status"], "error")
                self.assertIn("hash mismatch", result["error"])
                self.assertIsNone(result["reward"])


if __name__ == "__main__":
    unittest.main()
