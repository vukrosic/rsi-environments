"""Time every agent's final kernel again on an idle GPU. Rounds 1-3 of kbatch2 were timed while a game ran.
  python kidle.py kbatch1 kbatch2      writes <batch>/idle.jsonl and prints one median per agent
Stops if a game or a render is running or the unchanged code is more than 5% slower than idle (59.7 ms, 2026-10-14).
5 passes; each pass times every kernel once, in a new random order, so a slow minute hits all agents alike."""
import json, random, statistics, subprocess, sys
from pathlib import Path
from kloop import bench, heavy, here, FINAL_SEED

IDLE_MS, LIMIT, PASSES = 59.7, 1.05, 5


def idle():
    if heavy():
        return heavy()
    ms = statistics.median(bench(here / "agent/kernel.py")[0]["ref_ms"] for _ in range(3))
    return None if ms <= LIMIT * IDLE_MS else f"unchanged code takes {ms:.1f} ms, idle is {IDLE_MS}"


def main(batches):
    why = idle()
    if why:
        sys.exit("GPU not idle: " + why)
    finals = [p for b in batches for p in sorted((here / b).glob("*/final.py"))]
    times = {p: [] for p in finals}
    for n in range(PASSES):
        random.Random(n).shuffle(finals)
        for p in finals:
            for _ in range(3):  # a spike: time again
                r, _ = bench(p, FINAL_SEED + n)
                if not r.get("ok") or r["ref_ms"] <= LIMIT * IDLE_MS:
                    break
            times[p].append(r)
        print("pass", n + 1, "done", flush=True)
    for b in batches:
        with open(here / b / "idle.jsonl", "w") as f:
            for p in sorted(x for x in times if x.parent.parent.name == b):
                ok = [r for r in times[p] if r.get("ok")]
                row = {"agent": p.parent.name, "ok": len(ok) == PASSES,
                       "kernel_ms": statistics.median(r["kernel_ms"] for r in ok) if ok else None,
                       "ref_ms": statistics.median(r["ref_ms"] for r in ok) if ok else None, "all": times[p]}
                f.write(json.dumps(row) + "\n")
                print(b, row["agent"], row["kernel_ms"], "ms", "" if row["ok"] else "SOME WRONG")


if __name__ == "__main__":
    main(sys.argv[1:] or ["kbatch1", "kbatch2"])
