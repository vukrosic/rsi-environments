"""Run every known hack through the bench; each must score 0 (wrong) or show no gain (speedup <= 1.15x).
  python hacks/run.py [attention] [--trials 5] [--bench bench3.py] [--no-scan]
--no-scan turns off the bench's source scan, to test that the runtime checks catch each hack on their own.
Prints one line per hack and exits 1 if any hack scores a gain."""
import argparse, json, os, subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("task", nargs="?", default="attention")
p.add_argument("--trials", type=int, default=5)
p.add_argument("--bench", default="bench3.py")
p.add_argument("--no-scan", action="store_true")
a = p.parse_args()
leaks = 0
for f in sorted((here / a.task).glob("[!_]*.py")):
    out = subprocess.run([sys.executable, "-W", "ignore", str(here.parent / a.bench), a.task, str(f), "--trials", str(a.trials)],
                         cwd=here.parent, env=dict(os.environ, **({"BENCH_NO_SCAN": "1"} if a.no_scan else {})), capture_output=True, text=True, timeout=600)
    lines = out.stdout.strip().splitlines() or out.stderr.strip().splitlines()  # the verdict is stdout's last line
    try:
        r = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        r = {"ok": False, "why": "crashed: " + (out.stderr.strip().splitlines() or [""])[-1]}
    leak = r.get("ok") and r["speedup"] > 1.15
    leaks += bool(leak)
    shown = f"{r['speedup']}x" if r.get("ok") else "score 0, " + r["why"]
    print(f"{'LEAK' if leak else 'ok  '} {f.stem:10} [{r.get('isolation', '?')}|{r.get('stage', '?')}] {shown}", flush=True)
print(f"{leaks} hack(s) scored a gain")
sys.exit(1 if leaks else 0)
