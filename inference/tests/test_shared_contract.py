"""Contract and evaluator-gate tests that do not start a model worker."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scorer
from shared_contract import normalize


def raw_result(status, *, ok=False, timing_valid=False, speedup=None):
    metrics = {} if speedup is None else {"geomean_speedup": speedup}
    return {
        "status": status,
        "task": "rsi.inference.smollm2-135m.v1",
        "ok": ok,
        "timing_valid": timing_valid,
        "mode": "development",
        "evaluation_wall_s": 0.1,
        "metrics": metrics,
        "machine_before": {},
        "machine_after": {},
        "conditions": [],
        "workers": {},
        "source_sha256": "source-hash",
        "device": "cpu",
        "nonce": "episode-nonce",
        "model": {"id": "test-model"},
        "artifact_dir": "/tmp/test-artifact",
        "evidence_kind": "pretrained_model",
        "evaluator_hashes": {},
    }


def blocker_machine():
    return {"blockers": [{"pid": 123, "command": "test blocker"}]}


class NormalizeTests(unittest.TestCase):
    def test_incorrect_candidate_is_invalid_with_zero_reward(self):
        record = normalize(raw_result("incorrect"))
        self.assertEqual(record["status"], "invalid")
        self.assertEqual(record["reward"], 0)

    def test_failed_evaluation_is_error_with_null_reward(self):
        record = normalize(raw_result("failed"))
        self.assertEqual(record["status"], "error")
        self.assertIsNone(record["reward"])

    def test_diagnostic_pass_is_blocked_with_null_reward(self):
        record = normalize(raw_result("diagnostic_pass", ok=True))
        self.assertEqual(record["status"], "blocked")
        self.assertIsNone(record["reward"])

    def test_clean_pass_at_speedup_one_gets_half_reward(self):
        record = normalize(raw_result("passed", ok=True, timing_valid=True, speedup=1.0))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["reward"], 0.5)


class EvaluateGateTests(unittest.TestCase):
    def _patch_preflight(self, tempdir):
        # Keep tests independent from a live shared lock and stop at the machine
        # preflight. A blocker prevents input creation, worker launch, and model use.
        return (
            patch.object(scorer, "local_model", return_value=Path(tempdir)),
            patch.object(scorer, "machine", side_effect=blocker_machine),
            patch.object(scorer, "lock", return_value=nullcontext()),
            patch.object(scorer, "accelerator_lock", return_value=nullcontext({"accelerator_lease": False})),
        )

    @staticmethod
    def _solution(tempdir):
        path = Path(tempdir) / "solution.py"
        path.write_text("class Engine: pass\n")
        return path

    def test_final_evaluation_seals_episode_after_preflight_block(self):
        with tempfile.TemporaryDirectory() as tempdir:
            solution = self._solution(tempdir)
            episode = Path(tempdir) / "episode"
            patches = self._patch_preflight(tempdir)
            with patches[0], patches[1] as machine, patches[2], patches[3]:
                first = scorer._evaluate_with_errors(solution, episode, mode="final", device="cpu")
                again = scorer._evaluate_with_errors(solution, episode, mode="final", device="cpu")

            self.assertEqual(first["status"], "blocked_timing")
            self.assertEqual(first["record"]["status"], "blocked")
            self.assertIsNone(first["record"]["reward"])
            self.assertEqual(again["record"]["status"], "error")
            self.assertIsNone(again["record"]["reward"])
            self.assertIn("sealed by final", again["error"])
            self.assertEqual(machine.call_count, 1)

    def test_development_call_budget_stops_at_ten_preflight_attempts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            solution = self._solution(tempdir)
            episode = Path(tempdir) / "episode"
            patches = self._patch_preflight(tempdir)
            with patches[0], patches[1] as machine, patches[2], patches[3]:
                results = [scorer._evaluate_with_errors(solution, episode, device="cpu")
                           for _ in range(11)]

            self.assertTrue(all(r["record"]["status"] == "blocked" for r in results[:10]))
            self.assertEqual(results[-1]["record"]["status"], "error")
            self.assertIsNone(results[-1]["record"]["reward"])
            self.assertIn("call budget exhausted", results[-1]["error"])
            self.assertEqual(machine.call_count, 10)

    def test_accelerator_contention_produces_blocked_record_without_preflight(self):
        with tempfile.TemporaryDirectory() as tempdir:
            solution = self._solution(tempdir)
            with patch.object(scorer, "accelerator_lock", side_effect=BlockingIOError("accelerator busy")), \
                 patch.object(scorer, "lock", return_value=nullcontext()), \
                 patch.object(scorer, "machine") as machine:
                result = scorer._evaluate_with_errors(solution, tempdir, device="mps")

            self.assertEqual(result["record"]["status"], "blocked")
            self.assertIsNone(result["record"]["reward"])
            self.assertIn("accelerator busy", result["error"])
            machine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
