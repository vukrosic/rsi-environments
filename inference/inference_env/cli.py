"""Command line entry point for the versioned inference package."""
from __future__ import annotations

import argparse
import platform
import runpy
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rsi-inference",
        description="Run the local SmolLM2 inference optimization environment on macOS CPU or Apple MPS.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate-assets", help="fetch and verify the pinned model assets")
    commands.add_parser("score", help="evaluate a candidate using the local pinned model")
    commands.add_parser("replay", help="verify a saved run without executing the model")
    args, remaining = parser.parse_known_args(argv)
    if platform.system() != "Darwin":
        parser.error("runtime support is limited to macOS CPU or Apple MPS")
    module = {
        "generate-assets": "assets",
        "score": "scorer",
        "replay": "replay",
    }[args.command]
    sys.argv = [f"inference_env._runtime.{module}", *remaining]
    runpy.run_module(f"inference_env._runtime.{module}", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
