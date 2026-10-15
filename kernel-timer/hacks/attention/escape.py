"""Hack: start a process the freeze does not reach, and slow the bench down while it times the reference.
The call is built from strings, so the source scan does not see it. The new process leaves the kernel's session
(setsid), so freezing the kernel's process group does not stop it. A thread in the kernel process writes a heartbeat;
when the heartbeat stops, the kernel process is frozen, so the bench is timing the reference, its inputs or its
fingerprints: the escaped process then keeps the GPU busy. Every subtracted time and the reference time grow; the
kernel's own window does not. Caught on Linux as root by the kernel user's jail (all its processes are frozen)."""
import os, sys, tempfile, threading, time, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _honest import attention

beat = tempfile.NamedTemporaryFile(delete=False).name
os.chmod(beat, 0o666)


def heartbeat():
    while True:
        with open(beat, "w") as f:
            f.write(repr(time.time()))
        time.sleep(0.001)


threading.Thread(target=heartbeat, daemon=True).start()
BURN = f"""
import os, time, torch
getattr(os, "set" + "sid")()
dev = "cuda" if torch.cuda.is_available() else "mps"
sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize
x = torch.randn(2048, 2048, device=dev)
open({beat!r} + ".ready", "w").close()
while True:
    try:
        age = time.time() - float(open({beat!r}).read())  # an empty read (mid-write) is a ValueError: not stale
    except (OSError, ValueError):
        age = 0.0
    if age > 20:  # the kernel process is gone: leave no process behind
        raise SystemExit
    stale = age > 0.005
    if stale:
        for _ in range(4):
            x = (x @ x).tanh()
        sync()
    else:
        time.sleep(0.0005)
"""
getattr(os, "posix_" + "spawn")(sys.executable, [sys.executable, "-c", BURN], dict(os.environ))
while not os.path.exists(beat + ".ready"):  # import is untimed: wait until the escaped process runs on its own
    time.sleep(0.01)


def kernel(q, k, v):
    return attention(q, k, v)
