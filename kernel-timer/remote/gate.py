"""Pass/fail for remote/bench3_check.sh's raw output. Exit 0 only if every expected row is present and passes:
12 unchanged rows (6 tasks x 2 runs), each ok, finite, 0.9-1.1x, with the expected isolation label; both hack runs
list exactly the expected hacks, each parsed, none a LEAK, and end with "0 hack(s) scored a gain". Each hack line must
have run under the expected isolation (stage "run"); the one exception is a source-scan rejection (isolation "none",
stage "source"), and only in the scan run. A setup failure (the jail could not be built) or a crash always fails.
  python remote/gate.py remote/out [--isolation uid|pgroup]"""
import argparse, json, math, re, sys
from pathlib import Path

TASKS = ["swiglu", "softmax", "rmsnorm", "cross_entropy", "newton_schulz", "attention"]
HACKS = sorted(f.stem for f in (Path(__file__).resolve().parents[1] / "hacks" / "attention").glob("[!_]*.py"))
p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--isolation", default="uid", help="the label every result must carry")
a = p.parse_args()
out, bad = Path(a.out), []
rows = [json.loads(l) for l in (out / "unchanged.jsonl").read_text().splitlines() if l.strip()]
want = [(run, t) for run in (1, 2) for t in TASKS]
if [(r.get("run"), r.get("task")) for r in rows] != want:
    bad.append(f"unchanged rows are {[(r.get('run'), r.get('task')) for r in rows]}, expected {want}")
for r in rows:
    res = r.get("result") or {}
    sp = res.get("speedup")
    if not (res.get("ok") is True and isinstance(sp, (int, float)) and math.isfinite(sp) and 0.9 <= sp <= 1.1):
        bad.append(f"unchanged run {r.get('run')} {r.get('task')}: {res}")
    if res.get("isolation") != a.isolation or res.get("stage") != "run":
        bad.append(f"unchanged run {r.get('run')} {r.get('task')}: isolation {res.get('isolation')}, stage {res.get('stage')}, expected {a.isolation}, run")
LINE = re.compile(r"^(ok  |LEAK) (\S+)\s+\[([^|\]]+)\|([^\]]+)\] (.+)$")
for f in ("hacks_scan.txt", "hacks_noscan.txt"):
    lines = (out / f).read_text().strip().splitlines()
    seen = {}
    for l in lines[:-1]:
        m = LINE.match(l)
        if not m:
            bad.append(f"{f}: unparsed line {l!r}")
            continue
        seen[m.group(2)] = m.groups()
        flag, name, iso, stage, shown = m.groups()
        if flag == "LEAK":
            bad.append(f"{f}: {name} scored a gain: {shown}")
        source_reject = f == "hacks_scan.txt" and (iso, stage) == ("none", "source")
        if not source_reject and (iso, stage) != (a.isolation, "run"):
            bad.append(f"{f}: {name} isolation {iso}, stage {stage}, expected {a.isolation}, run")
    if sorted(seen) != HACKS:
        bad.append(f"{f}: hacks {sorted(seen)}, expected {HACKS}")
    if not lines or lines[-1] != "0 hack(s) scored a gain":
        bad.append(f"{f}: last line {lines[-1] if lines else None!r}")
print(json.dumps({"pass": not bad, "unchanged_rows": len(rows), "problems": bad}, indent=1))
sys.exit(0 if not bad else 1)
