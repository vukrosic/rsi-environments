"""Measure tasks.FASTEST on this GPU: time each no-effort version of every task against its library op with bench3.
  python tools/fastest_baseline.py [--tasks rmsnorm softmax] [--trials 10] [--out fastest.json]
Versions: the eager reference, torch.compile(reference), torch.compile(library op). Each is scored by bench3 as a
kernel against the library op (or the reference, for tasks without one). The fastest version that beats the library
op by more than --margin becomes the task's entry; paste the printed FASTEST into tasks.py."""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import tasks  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tasks", nargs="*", default=list(tasks.TASKS))
    a.add_argument("--trials", type=int, default=10)
    a.add_argument("--margin", type=float, default=0.02)
    a.add_argument("--out")
    a = a.parse_args()
    src, d, res = (HERE / "tasks.py").read_text(), Path(tempfile.mkdtemp()), {}
    for t in a.tasks:
        lib = t in tasks.BASELINE
        versions = {"compile_reference": f"torch.compile({t})"}
        if lib:
            versions |= {"reference": t, "compile_library": f"torch.compile(lib_{t})"}
        res[t] = {}
        for kind, expr in versions.items():
            f = d / f"{t}-{kind}.py"
            f.write_text(src + f"\n\nkernel = {expr}\n")
            r = subprocess.run([sys.executable, str(HERE / "bench3.py"), t, str(f), "--trials", str(a.trials), "--timeout", "900",
                                "--baseline", "library"], capture_output=True, text=True, cwd=HERE)
            v = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"ok": False, "why": r.stderr[-300:]}
            res[t][kind] = v.get("speedup") if v.get("ok") else None
            print(t, kind, res[t][kind], "" if v.get("ok") else v.get("why", "")[:120], file=sys.stderr)
    fastest = {}
    for t, r in res.items():
        best = max(((s, k) for k, s in r.items() if s), default=None)
        if best and best[0] > 1 + a.margin:
            fastest[t] = best[1]
    if a.out:
        Path(a.out).write_text(json.dumps({"speedups": res, "FASTEST": fastest}, indent=1))
    print("FASTEST = " + json.dumps(fastest, indent=4))


if __name__ == "__main__":
    main()
