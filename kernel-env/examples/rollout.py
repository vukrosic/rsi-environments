"""Offline environment compatibility rollout; no model weights are updated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_env import KernelEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("work/rollout"))
    parser.add_argument("--task", default="softmax")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    env = KernelEnvironment(args.task, args.workspace, profile="smoke", device="cpu",
                            seed=args.seed, max_steps=3)
    observation = env.reset()
    actions = [{"type": "test", "code": observation["code"]}, {"type": "submit"}]
    transitions = []
    for action in actions:
        transition = env.step(action)
        transitions.append(transition)
        if transition["terminated"] or transition["truncated"]:
            break
    print(json.dumps({"policy": "offline starter control; no weight update",
                      "actions": actions, "transitions": transitions}, indent=2))


if __name__ == "__main__":
    main()
