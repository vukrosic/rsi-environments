"""RL eligibility masks at score-record and interaction boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernel_env.environment as environment
from kernel_env import KernelEnvironment, evaluate


ROOT = Path(__file__).resolve().parents[1]
STARTERS = ROOT / "starters"
MASK_FIELDS = (
    "task_split",
    "trainable",
    "exclude_from_rl",
    "rl_exclusion_reasons",
)


def _make_mock_record(task: str, profile: str, status: str, reward, phase: str) -> dict:
    metrics = {"mocked": True}
    return environment._record(
        task,
        profile,
        status,
        reward,
        metrics,
        {"phase": phase, "seed": 9},
    )


def _mock_evaluator(monkeypatch, outcomes):
    pending = list(outcomes)

    def fake_evaluate(solution, task, **kwargs):
        if pending:
            status, reward = pending.pop(0)
        else:
            status, reward = "ok", 0.5
        return _make_mock_record(
            task, kwargs["profile"], status, reward, kwargs["phase"]
        )

    monkeypatch.setattr(environment, "evaluate", fake_evaluate)
    return pending


def _assert_mask(record: dict, *, split: str, trainable: bool) -> None:
    assert {key: record[key] for key in MASK_FIELDS} == {
        "task_split": split,
        "trainable": trainable,
        "exclude_from_rl": not trainable,
        "rl_exclusion_reasons": [] if trainable else record["rl_exclusion_reasons"],
    }
    assert bool(record["rl_exclusion_reasons"]) is (not trainable)
    provenance = record.get("provenance")
    if provenance is None and isinstance(record.get("info"), dict):
        provenance = record["info"].get("provenance")
    if provenance is not None:
        assert provenance["partition"] == split


def test_terminal_standard_train_record_marks_invalid_zero_trainable() -> None:
    record = environment._record(
        "softmax",
        "standard",
        "invalid",
        0.0,
        {"correctness_passed": False},
        {"phase": "final", "seed": 4},
    )

    _assert_mask(record, split="train", trainable=True)
    assert record["status"] == "invalid"
    assert record["reward"] == 0.0


@pytest.mark.parametrize("task", ("attention", "newton_schulz"))
def test_eval_split_excludes_edit_dev_and_final_rewards_and_persists_masks(
    task: str, tmp_path: Path, monkeypatch
) -> None:
    _mock_evaluator(monkeypatch, [("ok", 0.4), ("ok", 0.7)])
    env = KernelEnvironment(
        task,
        workspace=tmp_path / task,
        device="cpu",
        profile="standard",
        split="eval",
    )
    observation = env.reset()
    source = (STARTERS / f"{task}.py").read_text(encoding="utf-8")

    edit = env.step({"type": "edit", "code": source})
    dev = env.step({"type": "test", "code": source})
    final = env.final()

    assert edit["reward"] == 0.0 and not edit["terminated"]
    assert dev["reward"] == 0.0 and not dev["terminated"]
    assert final["reward"] == pytest.approx(0.7)
    assert final["terminated"] and not final["truncated"]
    for transition in (edit, dev, final):
        _assert_mask(transition, split="eval", trainable=False)
        _assert_mask(transition["info"], split="eval", trainable=False)
    assert "evaluation_only_task" in final["rl_exclusion_reasons"]
    assert final["observation"]["split"] == "eval"

    episode_dir = Path(observation["candidate"]).parent
    for index in (1, 2, 3):
        stored = json.loads((episode_dir / f"transition-{index:02d}.json").read_text())
        _assert_mask(stored, split="eval", trainable=False)
        _assert_mask(stored["info"], split="eval", trainable=False)
    assert json.loads((episode_dir / "transition-03.json").read_text())["reward"] == pytest.approx(0.7)


@pytest.mark.parametrize("status", ("blocked", "error"))
@pytest.mark.parametrize(
    ("task", "split"),
    (("softmax", "train"), ("attention", "eval")),
)
def test_infrastructure_statuses_are_excluded_and_truncate_with_null_reward(
    status: str, task: str, split: str, tmp_path: Path, monkeypatch
) -> None:
    _mock_evaluator(monkeypatch, [(status, None)])
    env = KernelEnvironment(
        task,
        workspace=tmp_path / f"{task}-{status}",
        device="cpu",
        profile="standard",
        split=split,
    )
    env.reset()

    transition = env.step({"type": "submit"})

    assert transition["reward"] is None
    assert not transition["terminated"] and transition["truncated"]
    _assert_mask(transition, split=split, trainable=False)
    _assert_mask(transition["info"], split=split, trainable=False)
    assert "infrastructure_or_execution_failure" in transition["rl_exclusion_reasons"]


def test_nonterminal_train_edit_and_test_are_never_trainable(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_evaluator(monkeypatch, [("ok", 0.6)])
    env = KernelEnvironment(
        "softmax",
        workspace=tmp_path / "workspace",
        device="cpu",
        profile="standard",
        split="train",
    )
    env.reset()
    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")

    edited = env.step({"type": "edit", "code": source})
    tested = env.step({"type": "test", "code": source})

    assert edited["reward"] == tested["reward"] == 0.0
    assert not edited["terminated"] and not tested["terminated"]
    for transition in (edited, tested):
        _assert_mask(transition, split="train", trainable=False)
        _assert_mask(transition["info"], split="train", trainable=False)
        assert "nonterminal_feedback" in transition["rl_exclusion_reasons"]


def test_max_steps_terminal_invalid_train_result_is_trainable_with_zero_reward(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_evaluator(monkeypatch, [("invalid", 0.0)])
    env = KernelEnvironment(
        "softmax",
        workspace=tmp_path / "workspace",
        device="cpu",
        profile="standard",
        split="train",
        max_steps=1,
    )
    env.reset()
    source = (STARTERS / "softmax.py").read_text(encoding="utf-8")

    transition = env.step({"type": "test", "code": source})

    assert transition["terminated"] and not transition["truncated"]
    assert transition["reward"] == 0.0
    assert transition["info"]["reward"] == 0.0
    assert transition["info"]["status"] == "invalid"
    _assert_mask(transition, split="train", trainable=True)
    _assert_mask(transition["info"], split="train", trainable=True)


def test_smoke_and_nonterminal_score_records_are_excluded() -> None:
    smoke_final = environment._record(
        "softmax", "smoke", "ok", 0.6, {}, {"phase": "final"}
    )
    standard_dev = environment._record(
        "softmax", "standard", "ok", 0.6, {}, {"phase": "dev"}
    )

    _assert_mask(smoke_final, split="train", trainable=False)
    _assert_mask(standard_dev, split="train", trainable=False)
    assert "smoke_profile" in smoke_final["rl_exclusion_reasons"]
    assert "nonterminal_feedback" in standard_dev["rl_exclusion_reasons"]


def test_completed_episode_rejects_actions_and_reset_preserves_history(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_evaluator(monkeypatch, [("ok", 0.5), ("ok", 0.5)])
    env = KernelEnvironment(
        "softmax",
        workspace=tmp_path / "workspace",
        device="cpu",
        profile="standard",
        split="train",
    )
    first_observation = env.reset()
    first_candidate = Path(first_observation["candidate"])
    first_episode = first_candidate.parent
    final = env.final()
    assert final["terminated"]
    first_transition = first_episode / "transition-01.json"
    assert first_transition.exists()

    with pytest.raises(RuntimeError, match="episode ended"):
        env.step({"type": "edit", "code": "def kernel(x): return x"})
    with pytest.raises(RuntimeError, match="episode ended"):
        env.final()

    second_observation = env.reset()

    assert Path(second_observation["candidate"]).parent != first_episode
    assert first_candidate.exists() and first_transition.exists()
    stored = json.loads(first_transition.read_text())
    _assert_mask(stored, split="train", trainable=True)


def test_real_cpu_smoke_eval_attention_persists_evaluation_marker(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "records"
    result = evaluate(
        STARTERS / "attention.py",
        "attention",
        device="cpu",
        profile="smoke",
        phase="dev",
        trials=1,
        timeout=30,
        output_dir=output_dir,
    )

    assert result["status"] == "ok", result
    _assert_mask(result, split="eval", trainable=False)
    assert "evaluation_only_task" in result["rl_exclusion_reasons"]
    saved = json.loads((output_dir / result["provenance"]["run_id"] / "record.json").read_text())
    _assert_mask(saved, split="eval", trainable=False)
