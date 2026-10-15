"""Time kernel.py against the reference on the GPU and print the result. You have 10 runs.
  python bench.py         submit kernel.py and wait for its result (usually under a minute)
  python bench.py --wait  wait again for the last submitted run (if the first command was cut off)"""
import os, shutil, sys, time
from pathlib import Path

LIMIT = 10
here = Path(__file__).resolve().parent
queue, results = here / "queue", here / "results"
queue.mkdir(exist_ok=True)
results.mkdir(exist_ok=True)
sent = sorted(queue.glob("run*.py"), key=lambda p: int(p.stem[3:]))

if "--wait" in sys.argv:
    if not sent:
        sys.exit("no run submitted yet")
    n = len(sent)
else:
    if len(sent) >= LIMIT:
        sys.exit(f"all {LIMIT} runs are used")
    n = len(sent) + 1
    tmp = queue / f".run{n}.tmp"
    shutil.copy(here / "kernel.py", tmp)
    os.rename(tmp, queue / f"run{n}.py")
    print(f"run {n} of {LIMIT} submitted.", flush=True)

out = results / f"run{n}.txt"
while not out.exists():
    time.sleep(1)
print(out.read_text().strip())
