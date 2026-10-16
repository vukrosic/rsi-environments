#!/usr/bin/env python3
"""Tests for tools/check_private_words.py using temporary git repositories."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parent / "check_private_words.py"


def run_checker_on(files: dict[str, bytes], words: str = "") -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp, tempfile.NamedTemporaryFile("w", suffix=".txt") as wf:
        wf.write(words)
        wf.flush()
        root = Path(tmp)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        return subprocess.run(
            [sys.executable, str(CHECKER), str(root)],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, PRIVATE_WORDS_FILE=wf.name),
        )


CLEAN_CODE = (
    b"from openai import OpenAI\n"
    b"client = AsyncOpenAI(api_key='x')\n"
    b"# OpenAI-compatible endpoint\n"
    b"dataset = 'openai/gsm8k'\n"
)


class CheckPrivateWordsTest(unittest.TestCase):
    def test_clean_repo_exits_zero(self) -> None:
        result = run_checker_on({"src/clean.py": CLEAN_CODE})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dirty_repo_exits_one_and_names_file_and_line(self) -> None:
        result = run_checker_on(
            {
                "src/clean.py": CLEAN_CODE,
                "notes/dirty.md": b"ok line\nsee /Users/alice/private.txt\n",
            }
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("notes/dirty.md:2", result.stdout)
        self.assertIn("/Users/", result.stdout)

    def test_words_file_patterns_are_caught_case_insensitively(self) -> None:
        words = "# comment line\nsecret-project\nCodename Blue\n"
        for sample in ["Secret-Project", "codename blue", "/Users/"]:
            with self.subTest(sample=sample):
                result = run_checker_on({"doc.md": f"line {sample} end\n".encode()}, words)
                self.assertEqual(result.returncode, 1, result.stdout)
        result = run_checker_on({"doc.md": b"comment line only\n"}, words)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_binary_files_are_skipped(self) -> None:
        result = run_checker_on({"blob.bin": b"\0\1/Users/x\0"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_files_over_five_mib_are_skipped(self) -> None:
        big = b"/Users/x\n" + b"a" * (5 * 1024 * 1024 + 1)
        result = run_checker_on({"huge.txt": big})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
