"""Adversarial action parsing and lifecycle-budget regressions."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import kernel_env.environment as environment
from kernel_env import KernelEnvironment


ROOT = Path(__file__).resolve().parents[1]
STARTERS = ROOT / "starters"


def _mock_evaluator(monkeypatch, calls=None):
    calls = [] if calls is None else calls

    def fake_evaluate(solution, task, **kwargs):
        calls.append(
            {
                "solution": Path(solution).read_text(encoding="utf-8"),
                "task": task,
                **kwargs,
            }
        )
        return environment._record(
            task,
            kwargs["profile"],
            "ok",
            0.5,
            {"mocked": True, "scored": True},
            {"phase": kwargs["phase"], "seed": kwargs["seed"]},
        )

    monkeypatch.setattr(environment, "evaluate", fake_evaluate)
    return calls


def _new_env(tmp_path: Path, *, task="softmax", **kwargs) -> KernelEnvironment:
    return KernelEnvironment(task, workspace=tmp_path / "workspace", **kwargs)


def _assert_schema_invalid(transition: dict, *, terminal: bool) -> None:
    assert transition["info"]["status"] == "invalid"
    assert transition["info"]["reward"] == 0
    assert transition["reward"] == 0
    assert transition["terminated"] is terminal
    assert transition["truncated"] is False
    assert transition["info"]["provenance"]["stage"] == "policy_schema"
    assert transition["info"]["metrics"]["scored"] is False
    assert transition["observation"]["feedback"]["provenance"]["stage"] == "policy_schema"


@pytest.mark.parametrize(
    "action",
    (
        None,
        7,
        False,
        [],
        {"type": "unknown"},
        {"type": "test", "code": "def kernel(x): return x", "extra": "forbidden"},
        {"type": "edit"},
        {"type": "edit", "code": ""},
        {"type": "edit", "code": "  \n \t"},
        {"type": "edit", "code": "def kernel(x): return x", "spare": True},
    ),
)
def test_malformed_actions_return_invalid_without_raising_or_evaluation(
    action, tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(
        tmp_path,
        profile="standard",
        max_steps=3,
    )
    observation = env.reset()
    candidate = Path(observation["candidate"])
    starter = candidate.read_bytes()

    transition = env.step(action)

    _assert_schema_invalid(transition, terminal=False)
    assert transition["observation"]["remaining_steps"] == 2
    assert candidate.read_bytes() == starter
    assert calls == []


def test_overlong_and_unencodable_sources_do_not_change_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(tmp_path, profile="standard", max_steps=4)
    candidate = Path(env.reset()["candidate"])
    starter = candidate.read_bytes()

    for code in ("x" * (2 * 1024 * 1024 + 1), "\ud800"):
        transition = env.step({"type": "edit", "code": code})
        _assert_schema_invalid(transition, terminal=False)
        assert candidate.read_bytes() == starter

    assert calls == []


def test_malformed_actions_consume_budget_and_last_one_terminates(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(tmp_path, profile="standard", max_steps=3)
    env.reset()

    first = env.step(None)
    second = env.step({"type": "edit"})
    final = env.step({"type": "submit", "code": ""})

    _assert_schema_invalid(first, terminal=False)
    _assert_schema_invalid(second, terminal=False)
    _assert_schema_invalid(final, terminal=True)
    assert final["info"]["provenance"]["phase"] == "final"
    assert final["info"]["provenance"]["stage"] == "policy_schema"
    assert final["info"]["metrics"]["scored"] is False
    assert final["observation"]["remaining_steps"] == 0
    assert calls == []

    with pytest.raises(RuntimeError, match="episode ended"):
        env.final()
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step({"type": "submit"})


@pytest.mark.parametrize(
    ("task", "split", "trainable"),
    (("softmax", "train", True), ("attention", "eval", False)),
)
def test_terminal_standard_parser_invalidity_has_split_eligibility(
    task: str, split: str, trainable: bool, tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(
        tmp_path,
        task=task,
        profile="standard",
        split=split,
        max_steps=1,
    )
    env.reset()

    transition = env.step({"type": "submit", "code": " \n"})

    _assert_schema_invalid(transition, terminal=True)
    assert transition["info"]["task_split"] == split
    assert transition["info"]["trainable"] is trainable
    assert transition["trainable"] is trainable
    assert transition["info"]["exclude_from_rl"] is (not trainable)
    assert transition["info"]["provenance"]["partition"] == split
    assert calls == []


def test_expired_episode_blocks_before_source_write(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(
        tmp_path,
        profile="standard",
        max_episode_seconds=5,
    )
    candidate = Path(env.reset()["candidate"])
    original = candidate.read_bytes()
    env.started = time.monotonic() - env.max_episode_seconds - 1

    transition = env.step({"type": "edit", "code": "def kernel(x): return x"})

    assert transition["info"]["status"] == "blocked"
    assert transition["reward"] is None
    assert not transition["terminated"] and transition["truncated"]
    assert candidate.read_bytes() == original
    assert calls == []


def test_constructor_rejects_invalid_action_and_episode_budgets(tmp_path: Path) -> None:
    for max_steps in (True, False, 0, -1, 1001, 1.5):
        with pytest.raises(ValueError, match="max_steps"):
            _new_env(tmp_path, max_steps=max_steps)
    for seconds in (True, False, 0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="max_episode_seconds"):
            _new_env(tmp_path, max_episode_seconds=seconds)
    for seed in (True, False, -1, 1_000_000_000, 2.5):
        with pytest.raises(ValueError, match="seed"):
            _new_env(tmp_path, seed=seed)


def test_boundary_seed_produces_valid_mocked_dev_step(tmp_path: Path, monkeypatch) -> None:
    calls = _mock_evaluator(monkeypatch)
    env = _new_env(
        tmp_path,
        profile="standard",
        split="train",
        seed=999_999_999,
        max_steps=2,
    )
    env.reset()

    transition = env.step({"type": "test"})

    assert transition["reward"] == 0
    assert not transition["terminated"] and not transition["truncated"]
    assert len(calls) == 1
    assert 0 <= calls[0]["seed"] < 1_000_000_000


def test_real_candidate_timeout_is_null_reward_and_excluded(
    tmp_path: Path,
) -> None:
    env = _new_env(
        tmp_path,
        profile="smoke",
        split="train",
        timeout=1,
        max_steps=1,
    )
    env.reset()
    infinite = "def kernel(x):\n    while True:\n        pass\n"

    transition = env.step({"type": "submit", "code": infinite})

    assert transition["info"]["status"] == "error"
    assert transition["reward"] is None
    assert not transition["terminated"] and transition["truncated"]
    assert transition["info"]["exclude_from_rl"] is True
    assert transition["exclude_from_rl"] is True
