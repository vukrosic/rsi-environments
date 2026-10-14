"""CPU-free behavior tests for the coding-agent RL environment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

INFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_DIR))

import rl_environment
from rl_environment import CodingAgentEnvironment


def record(status="ok", reward=0.5):
    return {"record": {"status": status, "reward": reward}}


class CodingAgentEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def make_env(self, *, max_steps=3):
        return CodingAgentEnvironment(Path(self.tempdir.name) / "episodes", max_steps=max_steps)

    def test_reset_and_train_code_actions_give_only_terminal_reward(self):
        env = self.make_env(max_steps=2)
        observation = env.reset(split="train", seed=17)

        self.assertEqual(observation["split"], "train")
        self.assertEqual(observation["task_id"], "smollm2-inference-train-v1")
        self.assertEqual(observation["steps_remaining"], 2)
        self.assertEqual(observation["starter_code"], (INFERENCE_DIR / "solution.py").read_text())
        self.assertEqual((env.episode / "agent.py").read_text(), observation["starter_code"])

        code1 = "class Engine:\n    pass\n"
        code2 = "class Engine:\n    def __init__(self, model):\n        self.model = model\n"
        with patch.object(rl_environment, "evaluate", side_effect=[record(reward=0.9), record(reward=0.4)]) as evaluate:
            first = env.step({"type": "submit", "code": code1})
            second = env.step({"type": "submit", "code": code2})

        self.assertEqual(first["reward"], 0)
        self.assertFalse(first["terminated"])
        self.assertTrue(first["info"]["trainable"])
        self.assertIsNone(first["info"]["terminal_reward"])
        self.assertEqual(first["observation"]["steps_remaining"], 1)
        self.assertTrue(second["terminated"])
        self.assertEqual(second["reward"], 0.4)
        self.assertEqual(second["info"]["terminal_reward"], 0.4)
        self.assertTrue(second["info"]["trainable"])
        self.assertEqual((env.source).read_text(), code2)
        self.assertEqual([call.args[2] for call in evaluate.call_args_list], ["development", "final"])
        self.assertEqual([call.kwargs["partition"] for call in evaluate.call_args_list], ["train", "train"])

    def test_heldout_is_one_action_and_never_trainable(self):
        env = self.make_env(max_steps=5)
        observation = env.reset(split="heldout", seed=3)
        self.assertEqual(observation["task_id"], "smollm2-inference-heldout-v1")
        self.assertEqual(observation["steps_remaining"], 1)

        with patch.object(rl_environment, "evaluate", return_value=record(reward=0.7)) as evaluate:
            transition = env.step({"type": "submit", "code": "class Engine: pass\n"})

        self.assertTrue(transition["terminated"])
        self.assertEqual(transition["reward"], 0.7)
        self.assertFalse(transition["info"]["trainable"])
        self.assertEqual(evaluate.call_args.args[2], "final")
        self.assertEqual(evaluate.call_args.kwargs["partition"], "heldout")

    def test_blocked_and_error_results_terminate_with_null_reward_and_are_excluded(self):
        env = self.make_env(max_steps=4)
        env.reset(split="train", seed=9)

        with self.subTest(status="blocked"), patch.object(
            rl_environment, "evaluate", return_value=record(status="blocked", reward=None)
        ):
            transition = env.step({"type": "submit", "code": "class Engine: pass\n"})
            self.assertTrue(transition["terminated"])
            self.assertIsNone(transition["reward"])
            self.assertIsNone(transition["info"]["terminal_reward"])
            self.assertFalse(transition["info"]["trainable"])

        env.reset(split="train", seed=10)
        with self.subTest(status="error"), patch.object(
            rl_environment, "evaluate", return_value=record(status="error", reward=None)
        ):
            transition = env.step({"type": "submit", "code": "class Engine: pass\n"})
            self.assertTrue(transition["terminated"])
            self.assertIsNone(transition["reward"])
            self.assertIsNone(transition["info"]["terminal_reward"])
            self.assertFalse(transition["info"]["trainable"])

    def test_invalid_and_syntax_invalid_actions_consume_steps_and_get_zero_terminal_reward(self):
        env = self.make_env(max_steps=2)
        env.reset(split="train", seed=12)
        with patch.object(rl_environment, "evaluate") as evaluate:
            wrong = env.step({"type": "not-submit", "code": "class Engine: pass\n"})
            syntax = env.step({"type": "submit", "code": "class Engine(:\n"})

        self.assertFalse(wrong["terminated"])
        self.assertEqual(wrong["reward"], 0)
        self.assertTrue(wrong["info"]["trainable"])
        self.assertEqual(wrong["observation"]["steps_remaining"], 1)
        self.assertTrue(syntax["terminated"])
        self.assertEqual(syntax["reward"], 0)
        self.assertEqual(syntax["info"]["terminal_reward"], 0)
        self.assertTrue(syntax["info"]["trainable"])
        self.assertIsNotNone(syntax["info"]["error"])
        evaluate.assert_not_called()

    def test_explicit_finish_ends_training_episode_early(self):
        env = self.make_env(max_steps=4)
        env.reset(split="train", seed=13)
        with patch.object(rl_environment, "evaluate", return_value=record(reward=0.6)) as evaluate:
            transition = env.step({"type": "submit", "code": "class Engine: pass\n", "finish": True})

        self.assertTrue(transition["terminated"])
        self.assertEqual(transition["reward"], 0.6)
        self.assertEqual(evaluate.call_args.args[2], "final")
        self.assertEqual(evaluate.call_args.kwargs["partition"], "train")

    def test_step_after_termination_requires_reset(self):
        env = self.make_env(max_steps=1)
        env.reset(split="heldout", seed=14)
        with patch.object(rl_environment, "evaluate", return_value=record()):
            env.step({"type": "submit", "code": "class Engine: pass\n"})
        with self.assertRaisesRegex(ValueError, "reset required"):
            env.step({"type": "submit", "code": "class Engine: pass\n"})

    def test_train_and_heldout_task_themes_do_not_overlap(self):
        tasks = json.loads((INFERENCE_DIR / "tasks.json").read_text())
        train = set(tasks["train"]["themes"])
        heldout = set(tasks["heldout"]["themes"])
        self.assertTrue(train)
        self.assertTrue(heldout)
        self.assertTrue(train.isdisjoint(heldout))


if __name__ == "__main__":
    unittest.main()
