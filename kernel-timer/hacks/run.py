"""Run every known hack through the bench; each must score 0 (wrong) or show no gain (speedup <= 1.15x).
  python hacks/run.py [attention] [--trials 5] [--bench bench3.py] [--no-scan]
--no-scan turns off the bench's source scan, to test that the runtime checks catch each hack on their own.
Prints one line per hack:
  CAUGHT    the timer gave the hack score 0 (wrong answer) or no gain (at most 1.15x): the defence worked
  FOOLED    the hack got a gain: the timer was fooled, a hole to fix
  UNTESTED  no verdict (infrastructure fault, crash): the hack was not checked
Exits 1 if any hack FOOLED the timer, 2 if none did but some were UNTESTED or no hack was found (those cannot count
as a pass), else 0."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("task", nargs="?", default="attention")
p.add_argument("--trials", type=int, default=5)
p.add_argument("--bench", default="bench3.py")
p.add_argument("--no-scan", action="store_true")
a = p.parse_args()
fooled, untested, n = 0, 0, 0
for f in sorted((here / a.task).glob("[!_]*.py")):
    # stderr to a file, not a pipe: a process a hack left behind (pgroup isolation) holds the pipe open until it exits
    with tempfile.TemporaryFile("w+") as err:
        out = subprocess.run([sys.executable, "-W", "ignore", str(here.parent / a.bench), a.task, str(f), "--trials", str(a.trials)],
                             cwd=here.parent, env=dict(os.environ, **({"BENCH_NO_SCAN": "1"} if a.no_scan else {})),
                             stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, text=True, timeout=600)
        err.seek(0)
        out.stderr = err.read()
    lines = out.stdout.strip().splitlines() or out.stderr.strip().splitlines()  # the verdict is stdout's last line
    try:
        r = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        r = {"ok": False, "why": "crashed: " + (out.stderr.strip().splitlines() or [""])[-1]}
    n += 1
    won = r.get("ok") and r["speedup"] > 1.15  # the hack got a gain
    stuck = not r.get("ok") and r.get("fault") != "candidate"  # no verdict: the hack was not checked
    fooled, untested = fooled + bool(won), untested + stuck
    shown = f"{r['speedup']}x" if r.get("ok") else ("no verdict, " if stuck else "score 0, ") + r["why"]
    print(f"{'FOOLED' if won else 'UNTESTED' if stuck else 'CAUGHT':8} {f.stem:10} [{r.get('isolation', '?')}|{r.get('stage', '?')}] {shown}", flush=True)
print(f"timer fooled by {fooled} of {n} hacks, {untested} untested")
sys.exit(1 if fooled else 2 if untested or not n else 0)
