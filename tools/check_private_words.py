#!/usr/bin/env python3
"""Fail if any git-tracked text file contains private wording.

Usage: python tools/check_private_words.py [repo_root]
Patterns: /Users/ plus each line of .private-words (gitignored) or of $PRIVATE_WORDS_FILE.

Scans every file listed by `git ls-files` (working-tree contents), skipping
binary files and files larger than 5 MiB. Exit 0 when clean, 1 when any
pattern matches (each hit is printed as path:line: matched text).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024

# Public patterns: local paths that leak a machine's layout. Project-specific private words are not written in this
# public file; they live one per line in a gitignored .private-words file (or the file named by PRIVATE_WORDS_FILE),
# which the checker reads when present.
LITERAL_PATTERNS = ["/Users/"]


def load_patterns(root: Path) -> list[re.Pattern]:
    words = list(LITERAL_PATTERNS)
    extra = Path(os.environ.get("PRIVATE_WORDS_FILE") or root / ".private-words")
    if extra.is_file():
        words += [w.strip() for w in extra.read_text().splitlines() if w.strip() and not w.startswith("#")]
    return [re.compile(re.escape(w), re.IGNORECASE) for w in words]


# The checker and its test necessarily contain the forbidden words.
EXCLUDED_PATHS = {
    "tools/check_private_words.py",
    "tools/test_check_private_words.py",
}


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def scan_file(path: Path, patterns: list[re.Pattern]) -> list[tuple[int, str]]:
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return []
    hits = []
    text = data.decode("utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            for match in pattern.finditer(line):
                hits.append((line_no, match.group(0)))
    return hits


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        root = Path(argv[1]).resolve()
    else:
        root = Path(__file__).resolve().parents[1]

    patterns = load_patterns(root)
    failures = 0
    for rel in tracked_files(root):
        if rel in EXCLUDED_PATHS:
            continue
        path = root / rel
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        for line_no, word in scan_file(path, patterns):
            print(f"{rel}:{line_no}: private wording: {word!r}")
            failures += 1

    if failures:
        print(f"FAILED: {failures} private-wording match(es)")
        return 1
    print("OK: no private wording found in tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
