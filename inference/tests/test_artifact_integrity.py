"""Input-bound artifact coverage and tensor integrity checks."""
from pathlib import Path
import sys
import unittest

import torch

INFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_DIR))

from scorer import compare, validate_artifact


def task_fixture():
    cases = []
    for key, length in (("case-a", 3), ("case-b", 5)):
        cases.append({
            "key": key,
            "ids": torch.tensor([[0] * length], dtype=torch.long),
            "forced": torch.tensor([[[0]], [[1]]], dtype=torch.long),
        })
    probes = {
        "a": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "b": torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
        "extension": torch.tensor([[1, 1]], dtype=torch.long),
        "branches": torch.tensor([[[0]], [[1]], [[0]]], dtype=torch.long),
        "batch": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "batch_tokens": torch.tensor([[1], [0]], dtype=torch.long),
    }
    return {"cases": cases, "decode_steps": 2, "vocab_size": 2,
            "cache_probe_inputs": probes}


def expected_probe_contexts(probes):
    a, b, branches = probes["a"], probes["b"], probes["branches"]
    return [
        ("request-a", a),
        ("request-a-step", torch.cat([a, branches[0]], dim=1)),
        ("request-b", b),
        ("request-a-after-b", a),
        ("branch-left-1", torch.cat([a, branches[0]], dim=1)),
        ("branch-left-2", torch.cat([a, branches[0], branches[1]], dim=1)),
        ("branch-right-reset", a),
        ("branch-right-1", torch.cat([a, branches[2]], dim=1)),
        ("repeated-prefix-extension", torch.cat([a, probes["extension"]], dim=1)),
        ("shorter-prefix-reset", a[:, :3]),
        ("batch-two-reset", probes["batch"]),
        ("batch-two-step", torch.cat([probes["batch"], probes["batch_tokens"]], dim=1)),
        ("batch-one-after-two", a),
    ]


def make_artifact(task):
    rows = []
    for case in task["cases"]:
        batch = case["ids"].shape[0]
        first = torch.tensor([[0.0, 1.0]] * batch)
        greedy = torch.tensor([[[1.0, 0.0]] * batch, [[0.0, 1.0]] * batch])
        forced = torch.tensor([[[0.0, 1.0]] * batch, [[1.0, 0.0]] * batch])
        tokens = torch.tensor([[[1]] * batch, [[0]] * batch], dtype=torch.long)
        rows.append({"key": case["key"], "length": case["ids"].shape[1],
                     "first": first, "greedy": greedy, "forced": forced, "tokens": tokens})
    cache_probes = [
        {"key": key, "context": context.clone(),
         "logits": torch.tensor([[0.0, 1.0]] * context.shape[0])}
        for key, context in expected_probe_contexts(task["cache_probe_inputs"])
    ]
    return {"rows": rows, "cache_probes": cache_probes}


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.task = task_fixture()

    def test_valid_artifact_covers_saved_inputs_and_all_thirteen_probes(self):
        artifact = make_artifact(self.task)
        self.assertIsNone(validate_artifact(artifact, self.task))
        self.assertTrue(compare(artifact, artifact, self.task)["ok"])

    def test_identical_artifacts_without_cache_probes_are_rejected(self):
        artifact = make_artifact(self.task)
        del artifact["cache_probes"]
        verdict = compare(artifact, artifact, self.task)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reward"], 0)

    def test_identical_artifacts_with_truncated_vocabulary_are_rejected(self):
        artifact = make_artifact(self.task)
        artifact["rows"][0]["first"] = artifact["rows"][0]["first"][:, :1]
        verdict = compare(artifact, artifact, self.task)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reward"], 0)

    def test_wrong_decode_steps_are_rejected(self):
        artifact = make_artifact(self.task)
        artifact["rows"][0]["greedy"] = artifact["rows"][0]["greedy"][:1]
        with self.assertRaises(ValueError):
            validate_artifact(artifact, self.task)

    def test_nonfinite_logits_are_rejected(self):
        artifact = make_artifact(self.task)
        artifact["cache_probes"][4]["logits"][0, 1] = float("nan")
        verdict = compare(artifact, artifact, self.task)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reward"], 0)

    def test_identical_artifacts_reject_bad_probe_identity_context_or_vocabulary(self):
        for change in ("context", "duplicate_key", "vocabulary"):
            with self.subTest(change=change):
                artifact = make_artifact(self.task)
                if change == "context":
                    artifact["cache_probes"][0]["context"][0, 0] = 1
                elif change == "duplicate_key":
                    artifact["cache_probes"][1]["key"] = artifact["cache_probes"][0]["key"]
                else:
                    artifact["cache_probes"][0]["logits"] = artifact["cache_probes"][0]["logits"][:, :1]
                verdict = compare(artifact, artifact, self.task)
                self.assertFalse(verdict["ok"])
                self.assertEqual(verdict["reward"], 0)

    def test_tokens_must_match_first_and_prior_greedy_argmax(self):
        artifact = make_artifact(self.task)
        artifact["rows"][0]["tokens"][1, 0, 0] = 1
        with self.assertRaises(ValueError):
            validate_artifact(artifact, self.task)

    def test_case_order_and_identity_are_bound_to_saved_inputs(self):
        artifact = make_artifact(self.task)
        artifact["rows"][1] = artifact["rows"][0]
        verdict = compare(artifact, artifact, self.task)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reward"], 0)

    def test_duplicate_saved_input_case_keys_are_rejected(self):
        task = task_fixture()
        task["cases"][1]["key"] = task["cases"][0]["key"]
        with self.assertRaises(ValueError):
            validate_artifact(make_artifact(task), task)


if __name__ == "__main__":
    unittest.main()
