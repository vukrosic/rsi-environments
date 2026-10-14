"""Actual scorer -> record -> RL transitions using synthetic worker artifacts."""
from pathlib import Path
import sys
import tempfile
import os
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorer
import rl_environment
from rl_environment import CodingAgentEnvironment


def task_fixture():
    cases = []
    for length in scorer.LENGTHS:
        cases.append({"key": f"case-{length}", "ids": torch.zeros((1, length), dtype=torch.long),
                      "forced": torch.tensor([[[0]], [[1]]], dtype=torch.long)})
    probes = {
        "a": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "b": torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
        "extension": torch.tensor([[1, 1]], dtype=torch.long),
        "branches": torch.tensor([[[0]], [[1]], [[0]]], dtype=torch.long),
        "batch": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "batch_tokens": torch.tensor([[1], [0]], dtype=torch.long),
    }
    return {"cases": cases, "decode_steps": 1, "vocab_size": 2,
            "partition": "train", "task_id": "fixture-train",
            "cache_probe_inputs": probes}


def probe_rows(task, offset=0.0):
    probes = task["cache_probe_inputs"]
    a, b, branches = probes["a"], probes["b"], probes["branches"]
    expected = [
        ("request-a", a),
        ("request-a-step", torch.cat([a, branches[0]], 1)),
        ("request-b", b),
        ("request-a-after-b", a),
        ("branch-left-1", torch.cat([a, branches[0]], 1)),
        ("branch-left-2", torch.cat([a, branches[0], branches[1]], 1)),
        ("branch-right-reset", a),
        ("branch-right-1", torch.cat([a, branches[2]], 1)),
        ("repeated-prefix-extension", torch.cat([a, probes["extension"]], 1)),
        ("shorter-prefix-reset", a[:, :3]),
        ("batch-two-reset", probes["batch"]),
        ("batch-two-step", torch.cat([probes["batch"], probes["batch_tokens"]], 1)),
        ("batch-one-after-two", a),
    ]
    return [{"key": key, "context": context.clone(),
             "logits": torch.tensor([[0.0, 1.0]] * context.shape[0]) + offset}
            for key, context in expected]


def artifact(duration, offset=0.0, task=None):
    cases = task["cases"] if task else [
        {"key": str(length), "ids": torch.zeros((1, length), dtype=torch.long),
         "forced": torch.tensor([[[0]], [[1]]], dtype=torch.long)}
        for length in scorer.LENGTHS
    ]
    rows = []
    for case in cases:
        logits = torch.tensor([[0.0, 1.0]]) + offset
        rows.append(dict(key=case["key"], length=case["ids"].shape[1], first=logits,
            greedy=logits.unsqueeze(0), forced=logits.unsqueeze(0).repeat(2, 1, 1),
            tokens=torch.tensor([[[1]]]), prefill_s=duration, decode_s=duration))
    result = dict(rows=rows, engine_init_s=0.001, warmup_s=0.001,
                  mps_memory={}, torch_threads=2)
    if task:
        result["cache_probes"] = probe_rows(task, offset)
    return result


def fixture_workers(task, oracle_offset=0.0, candidate_offset=0.0, malformed=None):
    def worker(source, inputs, output, device, deadline):
        role = output.stem
        shift = oracle_offset if role == "baseline_after" else (
            candidate_offset if role == "candidate" else 0.0)
        value = artifact(0.025 if role == "candidate" else 0.1, shift, task)
        if malformed == role:
            value["cache_probes"] = value["cache_probes"][:-1]
        if malformed == "candidate_rows" and role == "candidate":
            value["rows"] = []
            value.pop("cache_probes", None)
            value.pop("prefill_s", None)
            value.pop("decode_s", None)
        if malformed == "baseline_after_rows" and role == "baseline_after":
            value["rows"] = []
            value.pop("cache_probes", None)
        torch.save(value, output)
        return {"conditions": [], "artifact_sha256": scorer.sha(output)}
    return worker


class OracleExclusionTests(unittest.TestCase):
    def transition(self, oracle_offset=0.0, candidate_offset=0.0, max_steps=1, malformed=None):
        task = task_fixture()
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, RSI_INFERENCE_MEASUREMENT_LOCK=str(Path(root) / "measurement.lock")), \
             patch.object(rl_environment, "evaluate", side_effect=scorer._evaluate_with_errors), \
             patch.object(scorer, "local_model"), \
             patch.object(scorer, "machine", return_value={"blockers": []}), \
             patch.object(scorer, "make_inputs", return_value=task), \
             patch.object(scorer, "child", side_effect=fixture_workers(task, oracle_offset,
                                                                         candidate_offset, malformed)):
            env = CodingAgentEnvironment(root, max_steps=max_steps)
            observation = env.reset(split="train", seed=19)
            transition = env.step({"type": "submit", "code": observation["starter_code"]})
            if oracle_offset or malformed in ("baseline_before", "baseline_after_rows", "candidate_rows"):
                import json
                raw = json.loads(next(env.episode.glob("*/result.json")).read_text())
                self.assertFalse(raw["timing_valid"])
                if oracle_offset:
                    self.assertEqual(raw["status"], "failed")
                    self.assertIsNone(raw["reward"])
                    self.assertFalse(raw["metrics"]["baseline_comparison"]["ok"])
                    self.assertEqual(raw["metrics"]["candidate_matches_initial_reference"], candidate_offset == 0)
                elif malformed in ("baseline_before", "baseline_after_rows"):
                    self.assertEqual(raw["status"], "failed")
                    self.assertIsNone(raw["reward"])
                    self.assertIn("artifact incomplete", raw["error"])
                else:
                    self.assertEqual(raw["status"], "incorrect")
                    self.assertEqual(raw["reward"], 0)
                    self.assertNotIn("segments", raw.get("metrics", {}))
            return transition

    def test_consistent_oracle_keeps_correct_reward(self):
        transition = self.transition()
        self.assertEqual(transition["info"]["record"]["status"], "ok")
        self.assertAlmostEqual(transition["reward"], 0.8)
        self.assertTrue(transition["info"]["trainable"])

    def test_disagreement_excludes_matching_candidate(self):
        transition = self.transition(oracle_offset=1.0)
        self.assertIsNone(transition["reward"])
        self.assertFalse(transition["info"]["trainable"])
        self.assertTrue(transition["terminated"])
        self.assertEqual(transition["info"]["record"]["status"], "error")

    def test_disagreement_ends_nonterminal_episode(self):
        transition = self.transition(oracle_offset=1.0, max_steps=3)
        self.assertTrue(transition["terminated"])
        self.assertIsNone(transition["reward"])
        self.assertFalse(transition["info"]["trainable"])

    def test_disagreement_takes_precedence_over_bad_candidate(self):
        transition = self.transition(oracle_offset=1.0, candidate_offset=2.0)
        self.assertEqual(transition["info"]["record"]["status"], "error")
        self.assertIsNone(transition["reward"])
        self.assertFalse(transition["info"]["trainable"])

    def test_bad_candidate_with_consistent_oracle_still_gets_zero(self):
        transition = self.transition(candidate_offset=2.0)
        self.assertEqual(transition["info"]["record"]["status"], "invalid")
        self.assertEqual(transition["reward"], 0)
        self.assertTrue(transition["info"]["trainable"])

    def test_malformed_baseline_is_excluded_as_oracle_error(self):
        transition = self.transition(malformed="baseline_before")
        self.assertEqual(transition["info"]["record"]["status"], "error")
        self.assertIsNone(transition["reward"])
        self.assertFalse(transition["info"]["trainable"])

    def test_malformed_candidate_gets_invalid_zero_with_consistent_oracle(self):
        transition = self.transition(malformed="candidate")
        self.assertEqual(transition["info"]["record"]["status"], "invalid")
        self.assertEqual(transition["reward"], 0)
        self.assertTrue(transition["info"]["trainable"])

    def test_candidate_missing_rows_and_timings_gets_invalid_zero(self):
        transition = self.transition(malformed="candidate_rows")
        self.assertEqual(transition["info"]["record"]["status"], "invalid")
        self.assertEqual(transition["reward"], 0)
        self.assertTrue(transition["info"]["trainable"])

    def test_malformed_baseline_after_is_oracle_error(self):
        transition = self.transition(malformed="baseline_after_rows")
        self.assertEqual(transition["info"]["record"]["status"], "error")
        self.assertIsNone(transition["reward"])
        self.assertFalse(transition["info"]["trainable"])


if __name__ == "__main__":
    unittest.main()
