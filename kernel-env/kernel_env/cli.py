from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .environment import STARTERS, evaluate, calibrate
from .tasks import NAMES
from .difficulty import TRAIN_TASKS


def main():
    parser = argparse.ArgumentParser(description="Correctness-gated kernel optimization environment")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Copy an editable standalone starter (refuses overwrites)")
    init.add_argument("task", choices=NAMES)
    init.add_argument("path", type=Path)
    score = sub.add_parser("evaluate", help="Freeze candidate, check outputs, measure paired speed")
    score.add_argument("task", choices=NAMES)
    score.add_argument("solution", type=Path)
    score.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    score.add_argument("--phase", choices=("dev", "final"), default="dev")
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--trials", type=int)
    score.add_argument("--calibration", type=Path)
    cal = sub.add_parser("calibrate", help="Record fixed idle baselines; blocks under recognized load")
    cal.add_argument("task", choices=NAMES)
    cal.add_argument("--seed", type=int, default=2301)
    index = sub.add_parser("index-calibrations", help="Route one calibration path to existing task/family receipts; no measurements")
    index.add_argument("task", choices=TRAIN_TASKS)
    index.add_argument("--receipt", action="append", required=True, metavar="FAMILY=PATH")
    index.add_argument("--output", type=Path, required=True)
    for command in (score, cal):
        command.add_argument("--difficulty-id", help="Explicit training-only difficulty id; omit for the legacy workload")
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        command.add_argument("--timeout", type=float, default=180)
        command.add_argument("--lock-timeout", type=float, default=2)
        command.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "init":
        path = args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as target:
            target.write((STARTERS / f'{args["task"]}.py').read_text())
        print(json.dumps({"task": args["task"], "candidate": str(path.resolve())}))
        return
    if command == "index-calibrations":
        from .calibration_routes import write_index
        receipts = {}
        for entry in args["receipt"]:
            family, separator, path = entry.partition("=")
            if not separator or not family or not path or family in receipts:
                parser.error("--receipt needs unique FAMILY=PATH entries")
            receipts[family] = path
        path = write_index(args["output"], args["task"], receipts)
        print(json.dumps({"calibration_index": str(path), "measurements_created": False}))
        return
    result = evaluate(**args) if command == "evaluate" else calibrate(**args)
    print(json.dumps(result, indent=2, allow_nan=False))
    sys.exit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
