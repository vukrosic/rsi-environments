"""CPU-only check of bench3's Linux jail, as root: a process under the jail's uid that starts a grandchild in a new
session must be frozen, woken and killed with it, and a process of another uid must never be signalled.
  sudo python remote/jail_test.py      prints one JSON line; exit 0 only if every check passed"""
import json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench3

if sys.platform != "linux" or os.geteuid() != 0:
    print(json.dumps({"ok": False, "why": "needs Linux and root"}))
    sys.exit(2)


def state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return "gone"


def as_uid(uid):
    def pre():
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
    return pre


try:  # both constructors probe pidfd before anything is spawned
    j = bench3.Jail()
    other = bench3.Jail()  # a bystander with its own uid, standing in for an unrelated job
except RuntimeError as e:
    print(json.dumps({"ok": False, "why": str(e)}))
    sys.exit(1)
d = Path(tempfile.mkdtemp())
os.chmod(d, 0o777)
grand = d / "grand"
CHILD = f"""
import os, time
pid = os.posix_spawn("/bin/sh", ["/bin/sh", "-c", "exec setsid sleep 300"], dict(os.environ))
time.sleep(0.5)
open({str(grand)!r}, "w").write(str(pid))
while True:
    time.sleep(0.01)
"""
checks, bystander, child = {}, None, None
try:
    bystander = subprocess.Popen(["sleep", "300"], preexec_fn=as_uid(other.pw_uid))
    child = subprocess.Popen([sys.executable, "-c", CHILD], preexec_fn=as_uid(j.pw_uid), start_new_session=True)
    for _ in range(500):
        if grand.exists() and grand.read_text():
            break
        time.sleep(0.01)
    g = int(grand.read_text())
    checks["grandchild has the jail uid"] = os.stat(f"/proc/{g}").st_uid == j.pw_uid
    checks["grandchild left the child's session"] = os.getsid(g) != os.getsid(child.pid)
    j.signal(signal.SIGSTOP)
    time.sleep(0.2)
    checks["frozen: child"] = state(child.pid) == "T"
    checks["frozen: setsid grandchild"] = state(g) == "T"
    checks["bystander not frozen"] = state(bystander.pid) != "T"
    j.signal(signal.SIGCONT)
    time.sleep(0.2)
    checks["woken: child"] = state(child.pid) not in ("T", "gone")
    checks["woken: grandchild"] = state(g) not in ("T", "gone")
    j.signal(signal.SIGKILL)
    child.wait(timeout=5)
    time.sleep(0.2)
    checks["killed: grandchild"] = state(g) in ("gone", "Z")
    checks["no process left with the jail uid"] = all(state(q) == "Z" for q in j.pids(j.pw_uid))
    checks["bystander alive"] = bystander.poll() is None
except Exception as e:
    checks["ran to the end"] = False
    checks["error: " + f"{type(e).__name__}: {e}"[:200]] = False
finally:  # every failure still kills what this test started
    for jail in (j, other):  # pinned by pidfd, as in bench3: a reused pid is never signalled
        try:
            jail.signal(signal.SIGKILL)
        except RuntimeError:
            pass
    for pr in (child, bystander):
        if pr is not None:
            pr.kill()
            pr.wait()
    shutil.rmtree(d, ignore_errors=True)
ok = bool(checks) and all(checks.values())
print(json.dumps({"ok": ok, "checks": checks}))
sys.exit(0 if ok else 1)
