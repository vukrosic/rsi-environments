"""Behavioral tests for the inference artifact comparator."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer import compare


def artifact():
    """Small worker-shaped artifact with first, greedy, and forced logits."""
    return {
        "rows": [
            {
                "key": "case-0",
                "length": 3,
                "prefill_s": 0.01,
                "decode_s": 0.02,
                "first": torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]),
                "greedy": torch.tensor(
                    [
                        [[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]],
                        [[0.0, 2.0, 1.0], [1.0, 3.0, 2.0]],
                    ]
                ),
                "tokens": torch.tensor([[[2], [0]], [[1], [1]]]),
                "forced": torch.tensor(
                    [
                        [[0.5, 1.5, 2.5], [2.5, 1.5, 0.5]],
                        [[1.5, 0.5, 2.5], [0.5, 2.5, 1.5]],
                    ]
                ),
            }
        ]
    }


def reward(result):
    """The comparator's contract exposes the earned reward in its result."""
    return result["reward"]


class CompareTests(unittest.TestCase):
    def assert_rejected(self, candidate):
        result = compare(artifact(), candidate)
        self.assertFalse(result["ok"])
        self.assertEqual(reward(result), 0)

    def test_accepts_identical_artifacts(self):
        reference = artifact()
        self.assertTrue(compare(reference, copy.deepcopy(reference))["ok"])

    def test_accepts_logits_within_absolute_and_relative_tolerance(self):
        reference = artifact()
        candidate = copy.deepcopy(reference)
        # Below atol=.002, including near zero where relative tolerance adds little.
        candidate["rows"][0]["first"][0, 0] += 0.001
        candidate["rows"][0]["greedy"][0, 0, 0] += 0.001
        candidate["rows"][0]["forced"][0, 0, 0] += 0.001
        self.assertTrue(compare(reference, candidate)["ok"])

    def test_rejects_non_finite_logits(self):
        candidate = artifact()
        candidate["rows"][0]["forced"][1, 1, 2] = float("nan")
        self.assert_rejected(candidate)

    def test_rejects_wrong_logit_shape(self):
        candidate = artifact()
        candidate["rows"][0]["first"] = candidate["rows"][0]["first"][:, :2]
        self.assert_rejected(candidate)

    def test_rejects_greedy_token_mismatch_even_when_logits_match(self):
        candidate = artifact()
        candidate["rows"][0]["tokens"][0, 0, 0] = 1
        self.assert_rejected(candidate)

    def test_rejects_greedy_logits_outside_tolerance(self):
        candidate = artifact()
        candidate["rows"][0]["greedy"][1, 0, 0] += 0.01
        self.assert_rejected(candidate)

    def test_rejects_teacher_forced_logits_outside_tolerance(self):
        candidate = artifact()
        candidate["rows"][0]["forced"][0, 1, 1] -= 0.01
        self.assert_rejected(candidate)


if __name__ == "__main__":
    unittest.main()
