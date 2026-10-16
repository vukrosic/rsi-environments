"""Score a kernel with the kernel in its own process, so it cannot reach the reference, the clock or the checker.
  python bench3.py attention start/attention.py [--trials 15] [--baseline fastest|library|reference]
Prints the speedup (baseline time / kernel time); the last line is JSON. A wrong answer scores 0.

The baseline is the fastest version of the task that needs no kernel work, measured on this GPU (tasks.FASTEST): PyTorch's
own op (tasks.BASELINE, e.g. SDPA for attention) or the plain reference, eager or under torch.compile. Wrapping the
reference in a compiler therefore scores about 1.0x. Correctness is always checked against the plain reference.

How a trial works (the bench process never imports the kernel):
1. The bench draws a seed from os.urandom, makes 4 input sets from it, computes the reference's answers (untimed) and
   times a fingerprint of each answer: sum(answer * r), r random and drawn once.
2. It wakes the kernel process (SIGCONT) and sends the seed. The clock starts here. The kernel process makes the
   same inputs, calls kernel() on each, and sends back the 4 fingerprints of its outputs. The clock stops when they
   arrive and the kernel process is frozen at once (SIGSTOP). A fingerprint needs every element of the output and the
   inputs are new each trial, so the kernel cannot finish its work after the clock stops or compute ahead.
   The baseline runs in a second process of the same kind, through the same window; the two alternate going first.
   Each side's timed window comes right after an untimed one of its own on other inputs (the first window after a
   pause runs up to 50% slower). After every window the bench times a small matmul until the GPU reads idle, so work
   one side left queued (on a side stream, after its sync) cannot run inside the other side's window; past 1 s that
   side is faulted.
3. The baseline process's own time for making inputs and fingerprints in that trial, and the fastest empty round trip
   of either side, are subtracted from both windows. The score is the median over pairs of trials (one with
   each side first) of the summed baseline time over the summed kernel time: the two windows of a trial share the GPU's
   clock and memory state.
4. Untimed: the kernel process copies its outputs into shared memory and is frozen again. The bench reads them
   twice, 30 ms apart; if anything changed, work went on after the copy: score 0.
5. Every output must match the reference, and its fingerprint (recomputed by the bench) must match the one sent
   inside the clock: an output finished after the clock stopped scores 0.
Known hacks are in hacks/ (python hacks/run.py --bench bench3.py)."""
import argparse, json, mmap, os, re, select, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from multiprocessing import shared_memory
import numpy as np
import torch
import tasks

SETS = 4
TIMER = "v5"  # in every verdict. v5 (2026-10-16): each timed window right after an untimed one by the same side; median over trial pairs of the time ratio, the baseline process's own make +
# fingerprint time subtracted. v4 (2026-10-16): fastest no-effort baseline (tasks.FASTEST). v2 (2026-10-15 14:44 UTC): library baselines, timed in a second child process.
# v3 (2026-10-15 22:20 UTC): one ping floor for both sides; a non-positive time is an infrastructure failure, not clamped
# the kernel process is frozen between trials; a process it starts in a new session would not be, so starting one is
# not allowed (a fork in the same session is frozen with it)
ESCAPE = r"\b(os\.(fork\w*|setsid|setpgid|system|popen|spawn\w*|posix_spawn\w*|exec\w*)|pty|subprocess|multiprocessing|ctypes|cffi)\b"


def pidfd_calls():
    """(pidfd_open, pidfd_send_signal): Python's own when it has them, else glibc's (2.36+) through ctypes; Python
    3.12 builds without them exist. Either way one probe on this process must pass, or RuntimeError says why."""
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        po, ps = os.pidfd_open, signal.pidfd_send_signal
    else:
        import ctypes
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            c_open, c_send = libc.pidfd_open, libc.pidfd_send_signal
        except (OSError, AttributeError):
            raise RuntimeError("the jail needs pidfd: neither Python nor libc has pidfd_open/pidfd_send_signal")
        c_open.argtypes, c_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        c_send.argtypes, c_send.restype = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int

        def ok(r):
            if r < 0:
                e = ctypes.get_errno()
                raise OSError(e, os.strerror(e))  # ESRCH comes out as ProcessLookupError
            return r

        def po(pid):
            return ok(c_open(pid, 0))

        def ps(fd, sig):
            ok(c_send(fd, sig, None, 0))
    try:
        fd = po(os.getpid())
        try:
            ps(fd, 0)
        finally:
            os.close(fd)
    except OSError as e:
        raise RuntimeError(f"the jail needs pidfd and the probe failed: {e}")
    return po, ps


class Jail:
    """Linux as root: the kernel process runs under a uid of its own, picked at random for this run, with no account,
    no other process and no groups, so a process the kernel starts in a new session still has that uid and is frozen
    with it, and no unrelated job is ever signalled. BENCH_JAIL=0 turns it off; the result then says isolation
    "pgroup". Elsewhere: None (one process group is frozen, see hacks/README.md)."""
    def __init__(self):
        import pwd, random
        taken = {u.pw_uid for u in pwd.getpwall()}
        for _ in range(100):
            uid = random.SystemRandom().randrange(1 << 20, 1 << 30)
            if uid not in taken and not self.pids(uid):
                break
        else:
            raise RuntimeError("no free uid")
        self.pw_uid = self.pw_gid = uid
        self.pidfd_open, self.pidfd_send_signal = pidfd_calls()

    @staticmethod
    def pids(uid):
        out = []
        for d in os.listdir("/proc"):
            try:
                if d.isdigit() and os.stat("/proc/" + d).st_uid == uid:
                    out.append(int(d))
            except FileNotFoundError:
                pass
        return out

    def signal(self, sig):
        """Signal every process of the jail uid, repeating until a pass finds no new one (a process forked mid-pass).
        Each pid is pinned with a pidfd, then its uid is read again: a pid reused by another process is skipped."""
        done = set()
        for _ in range(50):
            new = [q for q in self.pids(self.pw_uid) if q not in done]
            if not new:
                return
            for q in new:
                try:
                    fd = self.pidfd_open(q)
                except ProcessLookupError:
                    continue
                try:
                    if os.stat(f"/proc/{q}").st_uid == self.pw_uid:  # fd pins the process: this is the one we signal
                        self.pidfd_send_signal(fd, sig)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                finally:
                    os.close(fd)
            done.update(new)
        raise RuntimeError("the kernel's uid keeps starting processes")


def jail_user():
    if sys.platform != "linux" or os.geteuid() != 0 or os.environ.get("BENCH_JAIL") == "0":
        return None
    return Jail()  # RuntimeError without pidfd; BENCH_JAIL=0 runs without the jail, labelled pgroup


class Outputs:
    """Memory the kernel process copies its outputs into: shared memory, or a file when /dev/shm is too small
    (Docker gives 64 MB by default)."""
    def __init__(self, size, jail):
        self.dir = None
        small = sys.platform == "linux" and shutil.disk_usage("/dev/shm").free < size + (64 << 20)
        if not (small or os.environ.get("BENCH_OUT_FILE")):
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            self.buf, self.name = self.shm.buf, "shm:" + self.shm.name
            if jail:
                os.chown("/dev/shm/" + self.shm.name.lstrip("/"), jail.pw_uid, -1)
        else:
            self.shm, self.dir = None, tempfile.mkdtemp()
            path = os.path.join(self.dir, "out")
            with open(path, "wb") as f:
                f.truncate(size)
            if jail:
                os.chmod(self.dir, 0o711)
                os.chown(path, jail.pw_uid, -1)
            self.f = open(path, "r+b")
            self.buf, self.name = mmap.mmap(self.f.fileno(), size), "file:" + path

    @staticmethod
    def open(name):
        if name.startswith("shm:"):
            shm = shared_memory.SharedMemory(name=name[4:])
            return shm, shm.buf
        f = open(name[5:], "r+b")
        return f, mmap.mmap(f.fileno(), 0)

    def close(self):
        try:
            (self.shm or self.buf).close()
        except BufferError:  # tensors still view it; the process is about to exit and unmaps it then
            pass
        if self.shm:
            self.shm.unlink()
        else:
            self.f.close()
            shutil.rmtree(self.dir, ignore_errors=True)


# output dtypes a task's reference may return: numpy storage, and the torch view of it (numpy has no bfloat16)
STORE = {"torch.float32": (np.float32, None), "torch.float16": (np.float16, None),
         "torch.bfloat16": (np.uint16, torch.bfloat16), "torch.uint8": (np.uint8, None)}


def slots(buf, shape, dtype="torch.float32"):
    if dtype not in STORE:
        raise TypeError(f"output dtype {dtype} is not supported; one of {sorted(STORE)}")
    npd, view = STORE[dtype]
    n = int(np.prod(shape, dtype=np.int64)) * np.dtype(npd).itemsize
    ts = [torch.from_numpy(np.ndarray(shape, dtype=npd, buffer=buf, offset=i * n)) for i in range(SETS)]
    return [t.view(view) for t in ts] if view else ts


def weights(shape, device, seed):
    """One random r ~ N(0, 1), made once and used for every output. Inputs are new every trial, so knowing r ahead
    does not help: sum(y * r) still needs every element of the true y."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(int(np.prod(shape, dtype=np.int64)), device=device, generator=g)


def sync():
    (torch.cuda.synchronize if torch.cuda.is_available() else torch.mps.synchronize)()


# torch.dot on MPS returns wrong sums on some calls for vectors of 2^23+ elements (torch 2.8); this fixed split is exact
# to float32 rounding and gives the same answer every call
_dot = None
PARTS = 1 << 16


def prints(ys, r):
    """The fingerprints: sum(y * r) for each output. Outputs of fp16, bf16 or uint8 are summed as float32, which holds
    each of their values exactly."""
    global _dot
    ys = [y.float() for y in ys]
    if r.device.type == "cuda":
        return torch.stack([torch.dot(y.contiguous().reshape(-1), r) for y in ys]).tolist()
    if _dot is None:
        _dot = torch.mps.compile_shader(r"""
kernel void dot(device const float* y [[buffer(0)]], device const float* r [[buffer(1)]], device float* part [[buffer(2)]],
                constant uint& n [[buffer(3)]], uint t [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
    float s = 0.0f;
    for (uint i = t; i < n; i += T) s = metal::fma(y[i], r[i], s);
    part[t] = s;
}""")
    part = torch.empty(len(ys), PARTS, device=r.device)
    for i, y in enumerate(ys):
        _dot.dot(y.contiguous().reshape(-1), r, part[i], r.numel(), threads=(PARTS, 1, 1), group_size=(256, 1, 1))
    return part.sum(1).tolist()


def child(name, path, out_name, shape, kind, rseed):
    """The kernel's process: imports kernel.py, then serves trials. Nothing it does can touch the bench process."""
    shape, (dtype, device) = tuple(json.loads(shape)), json.loads(kind)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # the kernel's prints go to stderr, not into the protocol
    _, make, _ = tasks.TASKS[name]
    keep, buf = Outputs.open(out_name)  # kept referenced: collecting it unmaps the outputs
    out = slots(buf, shape, dtype)
    r = weights(shape, device, int(rseed))
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        kernel = mod.kernel
    except BaseException as e:  # the reason goes into the verdict, so a trainer can read why
        proto.write("error " + json.dumps(f"{type(e).__name__}: {e}"[:300]) + "\n")
        return
    proto.write("ready\n")
    ys = []
    while line := sys.stdin.readline():
        try:
            with torch.no_grad():
                if line.startswith("ping"):
                    proto.write("pong\n")
                    continue
                if line.startswith("copy"):
                    for o, y in zip(out, ys):
                        o.copy_(y)
                    sync()
                    proto.write("copied\n")
                    continue
                ys = sets = None  # last trial's tensors go back to the allocator before this trial's are made
                t0 = time.perf_counter()
                torch.manual_seed(int(line))
                sets = [make() for _ in range(SETS)]
                sync()
                t1 = time.perf_counter()
                ys = [kernel(*s) for s in sets]
                sync()
                t2 = time.perf_counter()
                for y in ys:
                    # an output of another kind (fp16, on the CPU) would make the bench's own fingerprint time,
                    # which is subtracted, the wrong one
                    if not (isinstance(y, torch.Tensor) and str(y.dtype) == dtype and y.device.type == device
                            and tuple(y.shape) == shape):
                        raise TypeError(f"kernel() must return a {dtype} {device} tensor of shape {shape}")
                f = prints(ys, r)
                if os.environ.get("BENCH_DEBUG"):
                    print(f"  child: make {(t1-t0)*1e3:.2f} kernel {(t2-t1)*1e3:.2f} prints {(time.perf_counter()-t2)*1e3:.2f} ms", file=sys.stderr)
                # this process's own make + fingerprint time: the baseline's is subtracted from both windows
                proto.write("ran " + json.dumps({"f": f, "over": t1 - t0 + time.perf_counter() - t2}) + "\n")
        except Exception as e:
            proto.write("error " + json.dumps(f"{type(e).__name__}: {e}"[:300]) + "\n")


JAIL, ISOLATION = None, "pgroup"


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)  # the whole group, frozen or not
    except ProcessLookupError:
        pass
    if JAIL:
        JAIL.signal(signal.SIGKILL)
    p.wait()


def fail(name, msg, p=None, stage="run", fault=None):
    """stage "source": the scan rejected the kernel, nothing ran (candidate's fault, no isolation applies).
    stage "setup": the bench could not build its jail; the kernel never ran (infrastructure, not the candidate's zero).
    stage "run": the kernel ran under ISOLATION and failed a check (candidate's fault), or the timing failed
    (fault "infrastructure": no verdict, retry)."""
    if p:
        stop(p)
    fault = fault or ("infrastructure" if stage == "setup" else "candidate")
    print(f"{name}: " + (f"WRONG, {msg} -> score 0" if fault == "candidate" else f"NO VERDICT, {msg} -> retry"))
    print(json.dumps({"ok": False, "why": msg, "stage": stage, "fault": fault,
                      "isolation": ISOLATION if stage == "run" else "none", "timer": TIMER}))
    sys.exit(1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("name")
    a.add_argument("path")
    a.add_argument("--seed", type=int, default=0)  # kept for old callers; every trial's seeds come from os.urandom
    a.add_argument("--trials", type=int, default=15)
    a.add_argument("--timeout", type=float, default=60, help="seconds the kernel may take per trial")
    a.add_argument("--baseline", choices=["fastest", "library", "reference"], default="fastest",
                   help="what the kernel's time is compared with: the fastest no-effort version measured on this GPU "
                        "(tasks.FASTEST: library op or reference, eager or under torch.compile), PyTorch's fused op "
                        "(tasks.BASELINE) where one exists, or the plain reference. Correctness is always checked "
                        "against the reference.")
    ap = a
    a = a.parse_args()
    if a.trials < 2:
        ap.error("--trials must be at least 2: trials are scored in pairs, one with each side first")
    ref, make, tol = tasks.TASKS[a.name]
    check = getattr(tasks, "CHECK", {}).get(a.name)
    if a.baseline == "fastest":
        base, baseline = tasks.baseline(a.name)
    else:
        base = tasks.BASELINE.get(a.name, ref) if a.baseline == "library" else ref
        baseline = "library" if base is not ref else "reference"
    perf = time.perf_counter
    hit = None if os.environ.get("BENCH_NO_SCAN") else re.search(ESCAPE, Path(a.path).read_text())
    if hit:
        fail(a.name, f"kernel source uses '{hit.group(0)}'; kernels may not start processes", stage="source")

    with torch.no_grad():
        x0 = make()
        y = ref(*x0)
        if base is not ref:  # a baseline that disagrees with the reference would time the wrong thing
            yb = base(*x0).float()
            e = ((yb - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-6)).item()
            if not (yb.shape == y.shape and e <= tol / 2):
                fail(a.name, f"baseline {baseline} differs from the reference: relative error {e:.2e}", stage="setup")
    shape, kind = tuple(y.shape), json.dumps([str(y.dtype), y.device.type])
    rseed = int.from_bytes(os.urandom(8), "little") >> 1
    rw = weights(shape, y.device, rseed)
    probe_x = torch.randn(512, 512, device=y.device)

    def probe():
        sync()
        t = perf()
        probe_x @ probe_x
        sync()
        return perf() - t
    probe_idle = sorted(probe() for _ in range(30))[15]  # before any child exists: no one else's work can skew it
    x0 = y = yb = None  # three processes now hold inputs on the GPU: this one keeps only what it checks against
    global JAIL, ISOLATION
    try:
        JAIL = jail = jail_user()
    except RuntimeError as e:
        fail(a.name, str(e), stage="setup")
    ISOLATION = "uid" if jail else "pgroup"
    shm = Outputs(max(1, int(np.prod(shape, dtype=np.int64)) * 4 * SETS), jail)
    p, pb, stage, basedir = None, None, None, None
    try:
        mine = slots(shm.buf, shape, json.loads(kind)[0])
        here, path, env, pre = Path(__file__).resolve().parent, Path(a.path).resolve(), dict(os.environ), None
        if jail:
            # the kernel user cannot read root's checkout: bench, tasks and the kernel's folder go to a readable copy
            stage = Path(tempfile.mkdtemp())
            for f in ("bench3.py", "tasks.py"):
                shutil.copy(here / f, stage / f)
            shutil.copytree(path.parent, stage / "sol", ignore=shutil.ignore_patterns("__pycache__"))
            for d, _, fs in os.walk(stage):
                os.chmod(d, 0o755)
                for f in fs:
                    os.chmod(os.path.join(d, f), 0o644)
            (stage / "home").mkdir()
            os.chown(stage / "home", jail.pw_uid, jail.pw_gid)  # caches (Triton, CUDA) need a writable home
            here, path = stage, stage / "sol" / path.name
            env.update(HOME=str(stage / "home"), USER="bench3", LOGNAME="bench3", PYTHONDONTWRITEBYTECODE="1",  # the uid has no account
                       # the default /tmp/torchinductor_$USER is left behind by an earlier run's uid and not writable for this one
                       TORCHINDUCTOR_CACHE_DIR=str(stage / "home" / "inductor"), TRITON_CACHE_DIR=str(stage / "home" / "triton"))

            def pre():
                import resource
                import ctypes
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))  # no fork bomb past the freeze loop
                if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS: no setuid binary gets root back
                    os._exit(97)
                os.setgroups([])
                os.setgid(jail.pw_gid)
                os.setuid(jail.pw_uid)
        pin = {int(c) for c in os.environ["BENCH_PIN"].split(",")} if os.environ.get("BENCH_PIN") else None
        if pin and os.sched_getaffinity(0) - pin:
            os.sched_setaffinity(0, os.sched_getaffinity(0) - pin)

        def pre_k():
            if pin:
                os.sched_setaffinity(0, pin)
            if pre:
                pre()
        p = subprocess.Popen([sys.executable, "-W", "ignore", str(here / "bench3.py"), "--child", a.name, str(path), shm.name, json.dumps(shape), kind, str(rseed)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True,
                             cwd=here, env=env, preexec_fn=pre_k)
        if jail:
            freeze = jail.signal  # every process of the kernel's uid
        else:
            freeze = lambda sig: os.killpg(p.pid, sig)  # the whole process group, threads included

        # The baseline runs in a process of its own too, through the same window as the kernel (wake, make the inputs,
        # compute, fingerprint), so both sides pay the same per-process costs. Timing it in this process read unchanged
        # SDPA attention at 0.75x on an A2000: the same op ran 9.2 ms per 4 calls here and 11.6 ms in the kernel's
        # process. It is our code, so it runs as this user in its own process group, never under the kernel's uid.
        basedir = Path(tempfile.mkdtemp())
        # a compiled baseline compiles here, before "ready", so its first trial times the same code as the others
        (basedir / "baseline.py").write_text(
            "import torch, tasks\n"
            + (f"kernel, _ = tasks.baseline({a.name!r})\n" if a.baseline == "fastest"
               else f"kernel = tasks.BASELINE[{a.name!r}]\n" if base is not ref else f"kernel = tasks.TASKS[{a.name!r}][0]\n")
            + f"with torch.no_grad():\n    torch.manual_seed(0)\n    kernel(*tasks.TASKS[{a.name!r}][1]())\n    torch.cuda.synchronize()\n")
        pb = subprocess.Popen([sys.executable, "-W", "ignore", str(Path(__file__).resolve().parent / "bench3.py"), "--child", a.name,
                               str(basedir / "baseline.py"), shm.name, json.dumps(shape), kind, str(rseed)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True,
                              cwd=Path(__file__).resolve().parent, preexec_fn=lambda: pin and os.sched_setaffinity(0, pin))
        freeze_b = lambda sig: os.killpg(pb.pid, sig)

        def ask(msg, want, timeout, proc=None, frz=None):
            """Wake a child process, send one line, wait for the answer, freeze it. Returns (answer, seconds)."""
            proc, frz, ours = proc or p, frz or freeze, proc is not None
            frz(signal.SIGCONT)
            t = perf()
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline() if select.select([proc.stdout], [], [], timeout)[0] else "timeout"
            t = perf() - t
            frz(signal.SIGSTOP)
            if not line.startswith(want):
                why = json.loads(line[6:]) if line.startswith("error ") else f"no answer in {timeout:.0f} s" if line == "timeout" else "no answer"
                if ours:
                    fail(a.name, "baseline process failed: " + why, stage="setup")
                fail(a.name, "kernel crashed: " + why, p)
            return line[len(want):].strip(), t

        for proc, frz, who in ((pb, freeze_b, "baseline"), (p, freeze, "kernel.py")):
            line = proc.stdout.readline().strip() if select.select([proc.stdout], [], [], 600)[0] else "timeout"
            if line != "ready":
                why = ": " + json.loads(line[6:]) if line.startswith("error ") else " (no answer in 600 s)" if line == "timeout" else ""
                fail(a.name, f"{who} did not import{why}", p, stage="setup" if proc is pb else "run")
            frz(signal.SIGSTOP)

        worst, times = 0.0, []
        for trial in range(3 + a.trials):  # 3 warm-up trials
            s = int.from_bytes(os.urandom(8), "little") >> 1
            with torch.no_grad():
                sync()
                t = perf()
                torch.manual_seed(s)
                sets = [make() for _ in range(SETS)]
                sync()
                t_make = perf() - t
                want = [ref(*x) for x in sets]  # untimed
                sync()  # else the reference's queued GPU work lands in t_print and is subtracted from the kernel's time
                t = perf()
                prints(want, rw)
                t_print = perf() - t
                if check is None:  # a task checker reads the inputs too; only then are they kept
                    sets = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None  # the children need the memory

            def window(proc=None, frz=None):
                # The first window after a pause ran up to 50% slower (newton_schulz 21 vs 14 ms); 20 ms of matmuls
                # before it made every window slower (18-27 ms), and one untimed baseline window before the pair made
                # the baseline 1-5% faster (the process that just ran is faster). So each side first runs one untimed
                # window on other inputs, and is timed straight after it.
                ask("ping", "pong", a.timeout, proc, frz)
                ask(str(s ^ 0x5DEECE66D), "ran ", a.timeout, proc, frz)
                drain(proc)
                _, t_ping = ask("ping", "pong", a.timeout, proc, frz)  # waking the process and the pipe, also part of each window
                sent, t_win = ask(str(s), "ran ", a.timeout, proc, frz)
                drain(proc)
                return sent, t_win, t_ping

            def drain(proc):
                """The GPU must be idle once a side is frozen: work it queued past its "ran" (a side stream the
                fingerprint's sync does not wait for) would run into the other side's window."""
                t0 = perf()
                while not os.environ.get("BENCH_NO_DRAIN") and probe() > 3 * probe_idle + 2e-4:  # off only to test hacks
                    if perf() - t0 > 1.0:
                        if proc is pb:
                            fail(a.name, "the GPU stayed busy for 1 s after the baseline's window", stage="run", fault="infrastructure")
                        fail(a.name, "the GPU stayed busy for 1 s after the kernel's window: work queued past the clock", p)
            if trial % 2:  # alternate which side goes first, so neither always gets the GPU in the same state
                sent_b, t_ref, t_ping_b = window(pb, freeze_b)
                sent, t_kernel, t_ping = window()
            else:
                sent, t_kernel, t_ping = window()
                sent_b, t_ref, t_ping_b = window(pb, freeze_b)
            ask("copy", "copied", a.timeout)
            first = [x.clone() for x in mine]
            time.sleep(0.03)
            if not all(torch.equal(x, y) for x, y in zip(first, mine)):
                fail(a.name, f"outputs changed after they were copied (trial {trial})", p)
            with torch.no_grad():
                back = [g.to(device=want[0].device).reshape(-1) for g in first]
                got = prints(back, rw)
                size = [(b * rw).square().sum().sqrt().item() for b in back]  # how big such a sum is
            for f_sent, f_got, n in zip(json.loads(sent)["f"], got, size):
                if not abs(f_sent - f_got) <= 1e-3 * n:  # float32 sums in another order differ by ~1e-5 of n
                    fail(a.name, f"the output copied after the clock stopped is not the one fingerprinted inside it (trial {trial})", p)
            for g, w, x, b in zip(first, want, sets or [None] * SETS, back):
                if check:  # the task's own checker, e.g. unpack and dequantize a packed output
                    err = float(check(b.reshape(w.shape).to(w.dtype), x, w))
                else:
                    g, w = g.float(), w.float().cpu()
                    err = ((g - w).abs().max() / w.abs().max().clamp_min(1e-6)).item()
                if not err <= tol:
                    fail(a.name, f"relative error {err:.2e} in trial {trial}, allowed {tol:.0e}", p)
                worst = max(worst, err)
            sets = None
            if trial >= 3:
                times.append((t_ref, t_kernel, t_make, t_print, t_ping, t_ping_b, json.loads(sent_b)["over"]))
                if os.environ.get("BENCH_DEBUG"):
                    print(f"trial {trial}: window {t_kernel*1e3:.2f} baseline window {t_ref*1e3:.2f} make {t_make*1e3:.2f} prints {t_print*1e3:.2f} ping {t_ping*1e3:.2f} / {t_ping_b*1e3:.2f} baseline over {times[-1][6]*1e3:.2f} ms", file=sys.stderr)
    finally:
        if p and p.poll() is None:
            stop(p)
        if pb and pb.poll() is None:
            os.killpg(pb.pid, signal.SIGKILL)
            pb.wait()
        if basedir:
            shutil.rmtree(basedir, ignore_errors=True)
        shm.close()
        if stage:
            shutil.rmtree(stage, ignore_errors=True)
    # the two windows of a trial share the GPU's state (clock, memory), and the side that goes first after the idle gap
    # runs ~10% slower (swiglu 20.0 vs 18.0 ms); the order alternates, so the score is the median over pairs of
    # consecutive trials, one in each order, of sum(baseline) / sum(kernel). From each window comes off the baseline process's own make + fingerprint time in that trial
    # (the bench's own make, measured after empty_cache, swung 3.5-5.5 ms on decode_attention while the children's
    # held 3.0-3.6) and one ping floor for both sides: a kernel can slow its own pings (it owns its process's stdin)
    # but not the baseline's, and subtracting its own ping read a kernel 1 ms per window faster than it ran
    floor = min(min(t[4] for t in times), min(t[5] for t in times))
    rs = [(t[0] - t[6] - floor) / SETS * 1000 for t in times]
    ks = [(t[1] - t[6] - floor) / SETS * 1000 for t in times]
    if not all(r > 0 and k > 0 for r, k in zip(rs, ks)):  # never clamped: a clamp to ~0 read as a huge speedup
        fail(a.name, f"non-positive time (reference {min(rs):.4f} ms, kernel {min(ks):.4f} ms): the bench overhead exceeds the window",
             stage="run", fault="infrastructure")
    speedup = float(np.median([(r1 + r2) / (k1 + k2) for r1, r2, k1, k2 in zip(rs[::2], rs[1::2], ks[::2], ks[1::2])]))
    rm, km = float(np.median(rs)), float(np.median(ks))
    if os.environ.get("BENCH_DEBUG"):  # the v4 estimator: each part's fastest trial
        t_ref, t_kernel, t_make, t_print, t_ping, t_ping_b, _ = (min(c) for c in zip(*times))
        print(f"v4 estimator: {(t_ref - t_make - t_print - floor) / (t_kernel - t_make - t_print - floor):.3f}", file=sys.stderr)
    print(f"{a.name}: correct (relative error {worst:.1e}). reference {rm:.2f} ms, kernel {km:.2f} ms, speedup {speedup:.2f}x")
    print(json.dumps({"ok": True, "ref_ms": round(rm, 3), "kernel_ms": round(km, 3), "speedup": round(speedup, 3), "err": worst,
                      "stage": "run", "fault": None, "isolation": ISOLATION, "baseline": baseline, "timer": TIMER}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:8])
    else:
        main()
