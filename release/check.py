#!/usr/bin/env python3
"""One-command release check for the RSI environments repository.

    python release/check.py          run every check that can run on this machine
    python release/check.py --list   print the checks without running them
    python release/check.py -v       also print the output of passing checks

One line per check: PASS, FAIL or SKIP (with the reason). Then a summary.
Exit code 0 only if no check printed FAIL. Standard library only.

Unit tests run on CPU only (CUDA hidden, and the suites mock their devices).
Checks that need a CUDA GPU are skipped when no CUDA GPU is present.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

PYTEST = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"]

# (name, cwd relative to ROOT, command, timeout seconds, modules the suite imports)
UNIT_SUITES = [
    ("unit: shared records + leases", ".", PYTEST + ["suite/tests"], 300, ["pytest"]),
    ("unit: kernel-env", "kernel-env", PYTEST + ["tests"], 600, ["pytest", "torch", "numpy"]),
    ("unit: training-env", "training-env", PYTEST + ["tests"], 600, ["pytest", "torch", "numpy"]),
    ("unit: inference", "inference", PYTEST + ["tests"], 600, ["pytest", "torch", "numpy"]),
    ("unit: framework adapter", "suite/integration", PYTEST + ["tests"], 600, ["pytest", "verifiers"]),
    ("unit: kernel-timer report tool", "kernel-timer", PYTEST + ["tools/test_aa_report.py"], 120, ["pytest"]),
    ("unit: private-wording scanner", ".", [PY, "tools/test_check_private_words.py"], 120, []),
    ("hygiene: no private wording in tracked files", ".", [PY, "tools/check_private_words.py"], 120, []),
]

# Evidence files for the five release tests. Paths are relative to the repository root.
EVIDENCE = [
    ("test 1 same-task gain", [
        "experiments/2026-10-14-notes-attention/README.md",
        "experiments/2026-10-14-notes-attention/base-r1.json",
        "experiments/2026-10-14-notes-attention/notes-r1.md",
        "experiments/2026-10-14-notes-attention/chart.png",
    ]),
    ("test 2 transfer", [
        "experiments/2026-10-14-notes-transfer-cross-entropy/README.md",
        "experiments/2026-10-14-notes-transfer-cross-entropy/base.json",
        "experiments/2026-10-14-notes-transfer-newton-schulz/README.md",
        "experiments/2026-10-14-notes-transfer-newton-schulz/base.json",
        "experiments/2026-10-14-notes-transfer-newton-schulz/advice-generic.md",
    ]),
    ("test 3 tasks", [
        "kernel-timer/tasks.py",
        "kernel-timer/VALIDATION-V5.json",
        "kernel-timer/tools/fastest_baseline.py",
    ]),
    ("test 4 cheating", [
        "kernel-timer/hacks/run.py",
        "kernel-timer/hacks/attention/_honest.py",
        "kernel-timer/hacks/attention/cache.py",
        "kernel-timer/hacks/attention/fallback.py",
        "kernel-timer/hacks/attention/peek.py",
        "kernel-timer/hacks/attention/wrongref.py",
    ]),
    ("test 5 one command", [
        "release/check.py",
        "release/README.md",
    ]),
]


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cuda_gpu() -> tuple[bool, str]:
    """Return (present, detail). Uses nvidia-smi, then torch in a child process."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and "GPU" in r.stdout:
                return True, r.stdout.strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired):
            pass
    if sys.platform == "darwin":
        return False, "macOS has no CUDA"
    if has_module("torch"):
        try:
            r = subprocess.run([PY, "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return True, "torch.cuda available"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False, "no CUDA device found"


def run(cmd, cwd: Path, timeout: int, env_extra=None):
    """Run cmd in its own process group; kill the group on timeout. Returns (rc, output, seconds)."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = p.communicate()
        return None, (out or "") + f"\n[timed out after {timeout}s]", time.time() - t0


def repo_defines_module(name: str) -> bool:
    """True if the repository itself provides module `name` (a missing one is a real failure)."""
    for pattern in (f"**/{name}.py", f"**/{name}/__init__.py"):
        if any(".git" not in p.parts for p in ROOT.glob(pattern)):
            return True
    return False


def missing_third_party(output: str) -> str | None:
    for m in re.finditer(r"No module named '([\w.]+)'", output):
        top = m.group(1).split(".")[0]
        if not repo_defines_module(top):
            return top
    return None


def tail(text: str, n: int = 15) -> str:
    return "\n".join("      " + line for line in text.rstrip().splitlines()[-n:])


# Each check is a function returning (status, detail, output). Status: PASS / FAIL / SKIP.

def unit_check(cwd, cmd, timeout, modules):
    def go():
        for m in modules:
            if not has_module(m):
                return "SKIP", f"needs python module '{m}'", ""
        rc, out, secs = run(cmd, ROOT / cwd, timeout, {"CUDA_VISIBLE_DEVICES": ""})
        detail = f"{secs:.1f}s"
        if rc == 0:
            last = [l for l in out.strip().splitlines() if l.strip()]
            return "PASS", f"{last[-1] if last else 'ok'} ({detail})", out
        if rc is not None:
            dep = missing_third_party(out)
            if dep:
                return "SKIP", f"needs python module '{dep}'", out
        return "FAIL", ("timeout, " if rc is None else f"exit {rc}, ") + detail, out
    return go


def hacks_check():
    ok, detail = cuda_gpu()
    if not ok:
        return "SKIP", f"needs CUDA GPU ({detail})", ""
    if not has_module("torch"):
        return "SKIP", "needs python module 'torch'", ""
    rc, out, secs = run([PY, "hacks/run.py", "attention"], ROOT / "kernel-timer", 3600)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    if rc == 0:
        return "PASS", f"{last} ({secs:.0f}s)", out
    why = {1: "timer fooled", 2: "hacks untested or none found", None: "timeout"}.get(rc, f"exit {rc}")
    return "FAIL", f"{why}: {last} ({secs:.0f}s)", out


def evidence_check(paths):
    def go():
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            return "FAIL", f"{len(missing)} of {len(paths)} missing: {', '.join(missing)}", ""
        return "PASS", f"{len(paths)} files present", ""
    return go


def build():
    checks = []
    for name, cwd, cmd, timeout, modules in UNIT_SUITES:
        checks.append((name, f"{' '.join('python' if c == PY else c for c in cmd)} (in {cwd}, CPU only, {timeout}s limit)",
                       unit_check(cwd, cmd, timeout, modules)))
    checks.append(("timer hacks: every known fake scores no gain", "python hacks/run.py attention (in kernel-timer, needs CUDA GPU, 3600s limit)", hacks_check))
    for name, paths in EVIDENCE:
        checks.append((f"files exist (not a pass of the test): {name}", f"{len(paths)} files must exist", evidence_check(paths)))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print the checks without running them")
    ap.add_argument("-v", "--verbose", action="store_true", help="print output of passing checks too")
    args = ap.parse_args()
    checks = build()
    if args.list:
        for name, what, _ in checks:
            print(f"{name}\n    {what}")
        return 0
    width = max(len(n) for n, _, _ in checks)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for name, _, fn in checks:
        status, detail, out = fn()
        counts[status] += 1
        print(f"{status:4}  {name:{width}}  {detail}", flush=True)
        if out and (status == "FAIL" or args.verbose):
            print(tail(out, 25 if status == "FAIL" else 5), flush=True)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if counts["FAIL"]:
        print("RELEASE CHECK FAILED")
    elif counts["SKIP"]:
        print("no failures; skipped checks did not run on this machine")
    else:
        print("all checks passed")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
