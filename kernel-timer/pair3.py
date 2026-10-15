"""Paired timing for a kernel league (schema kernel-koth-paired-v1): the incumbent and the candidate are timed in the
same job, each in its own jailed process, alternating which goes first, with a frozen anchor timed in every pair.
  python pair3.py --describe                                  # runtime descriptor; runtime_sha256 = digest of it
  python pair3.py --build-request req.json --task rmsnorm --incumbent inc.py --candidate cand.py \
                  --interaction-seeds 1 2 --validation-seeds 3 4 [--anchor anchors/rmsnorm.py]
  python pair3.py --request req.json --output result.json [--lease-id ID | --lock PATH] [--job-id ID]
The last stdout line is the result JSON. Exit 0 ok, 2 invalid (the candidate failed), 3 infrastructure.

What it reuses from bench3: the child process (bench3.py --child), the uid jail, the freeze between windows, the
in-window fingerprint, the copy-then-recheck and the reference check. What is new:
1. Three members, each in a process of its own, under its own random uid when the jail is available: incumbent,
   candidate and anchor. The anchor is report-only (tasks.BASELINE where one exists, else the eager reference).
2. Each phase (interaction, then validation) is a list of pairs. Pair i uses the phase seed seeds[i % len(seeds)];
   its actual inputs come from that seed, the pair index and a nonce drawn from os.urandom for this job, so a member
   can neither compute ahead nor return an answer it cached from an earlier pair with the same seed. input_sha256 is
   the sha256 of the actual input tensors with their dtype, shape and device.
3. Within a pair the anchor goes first, then the two competitors: incumbent, candidate on even pairs, the other way
   on odd pairs. Each pair runs REPS rounds on fresh inputs; a member's time is its fastest round.
4. The GPU's idle speed is calibrated before any member starts; members then start one at a time. After every
   time a member is woken (its import, each window, each copy) the bench checks the GPU is idle again (a small probe
   matmul must run at its idle speed within 1 s). A frozen process's already-queued GPU work would keep running and
   slow the next member: that is the member's fault (hack G6, slowing the opponent), and the check right after its
   own turn puts the blame on it, not on whoever runs next.
   Before every window the bench runs 20 ms of its own GPU load (HEAT_S), so each member starts with the GPU in the
   same state. Without it the second member of a round ran ~2-4.5% faster on cross_entropy (A/A, A2000, pair3-v2).
   The SM clock of a row is read through NVML at the end of each preheat, while the GPU is still loaded; an
   nvidia-smi read after the round caught the GPU idling down (1200-1845 MHz on equal windows, pair3-v3).
   Warm-up (WARMUP rounds, untimed) alternates the competitors and puts the anchor last: with a fixed order the member
   warmed up last read 1-2.3% faster through the whole job, in both pair orders (rmsnorm A/A, pair3-v4).
5. Times are native: (window - the bench's own make, fingerprint and ping time) / SETS. The ping time is the fastest
   ping of any member in the phase, not the member's own ping in that round: one ping the OS delayed made a window
   negative (pair3-v3), and a member that slowed its own pings would have read faster. A non-positive time is a
   measurement failure (infrastructure), never clamped; such a receipt keeps the phase's raw rows as diagnostic_rows.
   Seeds: 2 to 50 distinct per phase; pair i uses seeds[i % len], so a list as long as the phase gives every pair
   its own seed. The job's nonce is published in the receipt after the run (input_nonce), so each pair's actual
   inputs can be rebuilt: torch seed = input_seed(nonce, "<phase>:<seed>", index, rep).
6. Failures: the candidate failing any check is "invalid" (blame candidate). The incumbent or anchor failing, a
   wrong request, a changed source, the wall or VRAM budget, or a bench error is "infrastructure" (null, not zero).
"""
import argparse, fcntl, hashlib, json, math, os, platform, re, shutil, signal, select, subprocess, sys, tempfile, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import tasks
from bench3 import SETS, ESCAPE, Outputs, slots, weights, sync, prints, jail_user

VERSION = "kernel-koth-paired-v1"
TIMER = "pair3-v5"
HEAT_S = 0.02
MODE = "same_job_shared_lease_alternating"
HERE = Path(__file__).resolve().parent
REPS = 3     # rounds per pair; a member's pair time is its fastest round
WARMUP = 4   # rounds per member before the first phase (compile, caches); recorded, not scored. Even: each competitor
# goes first in half of them
MEMBERS = ("incumbent", "candidate", "anchor")
perf = time.perf_counter


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(v):  # the same as suite.league.controller.digest
    return hashlib.sha256(canonical(v)).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smi(fields):
    """One nvidia-smi query of GPU 0 (the host has one GPU), or None."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return [x.strip() for x in out.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


class Nvml:
    """SM clock of GPU 0 through libnvidia-ml, cheap enough to read while the GPU is loaded; sm() is None without it."""
    def __init__(self):
        import ctypes
        self.c, self.h = ctypes, None
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            h = ctypes.c_void_p()
            if self.lib.nvmlInit_v2() == 0 and self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0:
                self.h, self.v = h, ctypes.c_uint()
        except (OSError, AttributeError):
            pass

    def sm(self):
        if self.h is None or self.lib.nvmlDeviceGetClockInfo(self.h, 1, self.c.byref(self.v)) != 0:  # 1 = NVML_CLOCK_SM
            return None
        return float(self.v.value)


def runtime():
    if torch.cuda.is_available():
        q = smi("uuid,name")
        gpu_uuid, name = (q[0], q[1]) if q else (None, torch.cuda.get_device_name(0))
        backend = "cuda"
    else:
        gpu_uuid, name, backend = f"mps:{platform.machine()}:{platform.mac_ver()[0]}", "apple-mps", "mps"
    return {"backend": backend, "gpu_uuid": gpu_uuid, "gpu_name": name, "torch": torch.__version__,
            "python": sys.version.split()[0], "timer": TIMER, "clock_mode": "observed", "reps": REPS, "warmup": WARMUP,
            "files": {f: sha(HERE / f) for f in ("pair3.py", "bench3.py", "tasks.py")}}


def anchor_file(task):
    """The frozen anchor source for a task, written once to anchors/<task>.py."""
    kind = "library_op" if task in tasks.BASELINE else "eager_reference"
    body = "import tasks\n" + (f"kernel = tasks.BASELINE[{task!r}]\n" if kind == "library_op" else f"kernel = tasks.TASKS[{task!r}][0]\n")
    path = HERE / "anchors" / f"{task}.py"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(body)
    return path, kind


def build_request(a, rt=None):
    rt = rt or runtime()
    anchor, kind = (Path(a.anchor), a.anchor_kind) if a.anchor else anchor_file(a.task)
    budget = {"interaction_trials": a.interaction_trials, "validation_trials": a.validation_trials,
              "wall_seconds": a.wall_seconds, "peak_vram_gb": a.peak_vram_gb}
    src = lambda p: {"path": str(Path(p).resolve()), "sha256": sha(p)}
    return {"schema_version": VERSION, "task": a.task, "usage": a.usage, "round_id": a.round_id, "timer": TIMER,
            "incumbent": src(a.incumbent), "candidate": src(a.candidate),
            "anchor": {**src(anchor), "kind": kind or "library_op", "role": "report_only"},
            "runtime_sha256": digest(rt), "gpu_uuid": rt["gpu_uuid"], "oracle_sha256": sha(HERE / "tasks.py"),
            "budget_sha256": digest(budget), "budget": budget,
            "interaction_seeds": a.interaction_seeds, "validation_seeds": a.validation_seeds,
            "measurement_mode": MODE, "min_retime_improvement": 0.02}


class Infra(Exception):
    pass


class Fault(Exception):
    """A member failed. stage: import, compile, correctness or runtime."""
    def __init__(self, role, stage, reason):
        super().__init__(f"{role}: {reason}")
        self.role, self.stage, self.reason = role, stage, reason


class Member:
    def __init__(self, role, path, jail, stage, task, shape, kind):
        self.role, self.jail = role, jail
        self.rseed = int.from_bytes(os.urandom(8), "little") >> 1
        self.out = Outputs(max(1, int(np.prod(shape, dtype=np.int64)) * 4 * SETS), jail)
        self.mine = slots(self.out.buf, shape, json.loads(kind)[0])
        self.rw = None
        home = stage / f"home-{role}"
        home.mkdir()
        env = dict(os.environ, HOME=str(home), PYTHONDONTWRITEBYTECODE="1",
                   TORCHINDUCTOR_CACHE_DIR=str(home / "inductor"), TRITON_CACHE_DIR=str(home / "triton"))
        pre = None
        if jail:
            os.chown(home, jail.pw_uid, jail.pw_gid)
            os.chmod(home, 0o700)
            env.update(USER="bench3", LOGNAME="bench3")

            def pre():
                import resource, ctypes
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
                if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
                    os._exit(97)
                os.setgroups([])
                os.setgid(jail.pw_gid)
                os.setuid(jail.pw_uid)
        self.p = subprocess.Popen([sys.executable, "-W", "ignore", str(stage / "bench3.py"), "--child", task, str(path),
                                   self.out.name, json.dumps(shape), kind, str(self.rseed)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True, cwd=stage, env=env, preexec_fn=pre)

    def freeze(self, sig):
        if self.jail:
            self.jail.signal(sig)  # every process of this member's uid
        else:
            os.killpg(self.p.pid, sig)

    def ready(self, timeout):
        line = self.p.stdout.readline().strip() if select.select([self.p.stdout], [], [], timeout)[0] else "timeout"
        if line != "ready":
            why = json.loads(line[6:]) if line.startswith("error ") else f"no answer in {timeout:.0f} s" if line == "timeout" else "process exited"
            raise Fault(self.role, "import", "did not import: " + why)
        self.freeze(signal.SIGSTOP)

    def ask(self, msg, want, timeout, stage="runtime"):
        """Wake the member, send one line, wait for the answer, freeze it. Returns (answer, seconds)."""
        self.freeze(signal.SIGCONT)
        t = perf()
        try:
            self.p.stdin.write(msg + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise Fault(self.role, stage, "process exited")
        line = self.p.stdout.readline() if select.select([self.p.stdout], [], [], timeout)[0] else "timeout"
        t = perf() - t
        self.freeze(signal.SIGSTOP)
        if not line.startswith(want):
            if line.startswith("error "):
                why = json.loads(line[6:])
                raise Fault(self.role, "correctness" if "must return" in why else stage, "kernel error: " + why)
            raise Fault(self.role, stage, f"no answer in {timeout:.0f} s" if line == "timeout" else "process exited")
        return line[len(want):].strip(), t

    def stop(self):
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.jail:
            self.jail.signal(signal.SIGKILL)
        self.p.wait()
        try:
            self.out.close()
        except FileNotFoundError:  # the child's resource tracker unlinked the shared memory when it was killed
            pass


def input_seed(nonce, phase, index, rep):
    return int.from_bytes(hashlib.sha256(f"{nonce}:{phase}:{index}:{rep}".encode()).digest()[:8], "little") >> 1


def hash_inputs(h, sets):
    for x in sets:
        for v in x:
            if isinstance(v, torch.Tensor):
                t = v.detach().contiguous()
                h.update(f"{t.dtype}|{tuple(t.shape)}|{t.device.type}|".encode())
                h.update(t.cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                h.update(repr(v).encode())


class Bench:
    def __init__(self, req, deadline, timeout):
        self.task = req["task"]
        self.ref, self.make, self.tol = tasks.TASKS[self.task]
        self.check = getattr(tasks, "CHECK", {}).get(self.task)
        self.deadline, self.timeout = deadline, timeout
        self.members, self.jails, self.stage = {}, [], None
        self.vram0 = self.vram_peak = None
        self.isolation = "pgroup"
        self.partial = None

    def left(self):
        t = self.deadline - time.monotonic()
        if t <= 0:
            raise Infra("wall budget reached")
        return t

    def setup(self, req):
        with torch.no_grad():
            torch.manual_seed(0)
            y = self.ref(*self.make())
        self.shape, self.kind = tuple(y.shape), json.dumps([str(y.dtype), y.device.type])
        self.dev = y.device
        self.nvml = Nvml()
        y = None
        q = smi("memory.used")
        self.vram0 = self.vram_peak = float(q[0]) if q else None
        self.stage = Path(tempfile.mkdtemp())
        for f in ("bench3.py", "tasks.py"):
            shutil.copy(HERE / f, self.stage / f)
        paths = {}
        for role in MEMBERS:
            d = self.stage / role
            d.mkdir()
            paths[role] = d / Path(req[role]["path"]).name
            shutil.copy(req[role]["path"], paths[role])
        for d, _, fs in os.walk(self.stage):
            os.chmod(d, 0o755)
            for f in fs:
                os.chmod(os.path.join(d, f), 0o644)
        try:
            for _ in MEMBERS:
                j = jail_user()
                if j is None:
                    break
                if any(j.pw_uid == k.pw_uid for k in self.jails):
                    raise Infra("jail uid collision")
                self.jails.append(j)
        except RuntimeError as e:
            raise Infra(str(e))
        self.isolation = "uid" if self.jails else "pgroup"
        self.rw = weights(self.shape, self.dev, 1)  # times the bench's own fingerprints
        self.probe_x = torch.randn(512, 512, device=self.dev)
        ts = sorted(self.probe() for _ in range(30))  # before any member exists: work queued at import can't skew it
        self.probe_idle = ts[len(ts) // 2]
        self.heat = torch.zeros(16 << 20, device=self.dev)
        imports = {}
        for i, role in enumerate(MEMBERS):  # one at a time, so GPU work left after an import has one owner
            t = perf()
            m = self.members[role] = Member(role, paths[role], self.jails[i] if self.jails else None, self.stage,
                                            self.task, self.shape, self.kind)
            m.rw = weights(self.shape, self.dev, m.rseed)
            m.ready(min(600, self.left()))
            imports[role] = round(perf() - t, 3)
            self.drain(role, "import")
        return imports

    def probe(self):
        sync()
        t = perf()
        self.probe_x @ self.probe_x
        sync()
        return perf() - t

    def preheat(self):
        """The bench's own memory + matmul load for HEAT_S, ending idle: every window starts from the same GPU state.
        Returns the SM clock read before the last sync, while the GPU is still loaded (None without NVML)."""
        t = perf()
        while True:
            self.heat.add_(1.0)
            self.probe_x @ self.probe_x
            if perf() - t >= HEAT_S:
                c = self.nvml.sm()
                sync()
                return c
            sync()

    def drain(self, role, stage="runtime"):
        """The GPU must be idle once a member is frozen; work it queued past its turn would slow the next member."""
        t0 = perf()
        while self.probe() > 3 * self.probe_idle + 2e-4:
            if perf() - t0 > 1.0:
                raise Fault(role, stage, f"the GPU stayed busy for 1 s after the member's turn ({stage}): work queued past the clock")

    def sample(self):
        q = smi("clocks.sm,memory.used")
        if not q:
            return None
        self.vram_peak = max(self.vram_peak, float(q[1]))
        return float(q[0])

    def rnd(self, s, order, h):
        """One round: make the inputs, time each member's window in order, then check every member's outputs.
        Returns each member's window and ping (seconds), the bench's make and fingerprint times, errors, load clocks."""
        with torch.no_grad():
            sync()
            t = perf()
            torch.manual_seed(s)
            sets = [self.make() for _ in range(SETS)]
            sync()
            t_make = perf() - t
            hash_inputs(h, sets)
            want = [self.ref(*x) for x in sets]
            sync()
            t = perf()
            prints(want, self.rw)
            t_print = perf() - t
            if self.check is None:
                sets = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        win, ping, sent, clocks = {}, {}, {}, []
        for role in order:
            m = self.members[role]
            c = self.preheat()
            if c is not None:
                clocks.append(c)
            _, ping[role] = m.ask("ping", "pong", min(self.timeout, self.left()))
            sent[role], win[role] = m.ask(str(s), "ran ", min(self.timeout, self.left()))
            self.drain(role)
        errs = {}
        for role in order:
            errs[role] = self.verify(role, sent[role], want, sets)
        return win, ping, t_make, t_print, errs, clocks

    def verify(self, role, sent, want, sets):
        m = self.members[role]
        m.ask("copy", "copied", min(self.timeout, self.left()))
        self.drain(role)
        first = [x.clone() for x in m.mine]
        time.sleep(0.03)
        if not all(torch.equal(x, y) for x, y in zip(first, m.mine)):
            raise Fault(role, "correctness", "outputs changed after they were copied")
        with torch.no_grad():
            back = [g.to(device=want[0].device).reshape(-1) for g in first]
            got = prints(back, m.rw)
            size = [(b * m.rw).square().sum().sqrt().item() for b in back]
        for f_sent, f_got, n in zip(json.loads(sent), got, size):
            if not abs(f_sent - f_got) <= 1e-3 * n:
                raise Fault(role, "correctness", "the output copied after the clock stopped is not the one fingerprinted inside it")
        worst = 0.0
        for g, w, x, b in zip(first, want, sets or [None] * SETS, back):
            if self.check:
                err = float(self.check(b.reshape(w.shape).to(w.dtype), x, w))
            else:
                g, w = g.float(), w.float().cpu()
                err = ((g - w).abs().max() / w.abs().max().clamp_min(1e-6)).item()
            if not err <= self.tol:
                raise Fault(role, "correctness", f"relative error {err:.2e}, allowed {self.tol:.0e}")
            worst = max(worst, err)
        return worst

    def warmup(self, nonce):
        """Untimed for the score: each member's window less its ping and the bench's own make + fingerprint time, ms.
        The competitors alternate and the anchor always goes last: the member warmed up last ran 1-2% faster for the
        rest of the job (rmsnorm A/A, A2000, pair3-v4), so that place goes to the report-only anchor."""
        out = {r: [] for r in MEMBERS}
        for i in range(WARMUP):
            try:
                win, ping, t_make, t_print, _, _ = self.rnd(input_seed(nonce, "warmup", i, 0),
                                                            (["incumbent", "candidate"] if i % 2 == 0 else ["candidate", "incumbent"]) + ["anchor"],
                                                            hashlib.sha256())
            except Fault as e:
                if e.stage == "runtime":
                    e.stage = "compile"
                raise
            for r in MEMBERS:
                out[r].append(round((win[r] - ping[r] - t_make - t_print) / SETS * 1000, 4))
        return out

    def phase(self, name, seeds, n, nonce, lease, gpu_uuid):
        """A member's pair time = (its fastest window in the pair's rounds - the bench's fastest make and fingerprint
        times in the phase - the fastest ping of any member in the phase) / SETS. The bench's own costs and the ping
        floor do not depend on the members, so their minimum over the phase is the best estimate of them (as bench3
        takes the minimum of each part); a member can slow its own pings but cannot make the floor lower."""
        pairs, makes, printing, pings = [], [], [], []
        for i in range(n):
            order = ["incumbent", "candidate"] if i % 2 == 0 else ["candidate", "incumbent"]
            h, best, ping, errs = hashlib.sha256(), {r: [] for r in MEMBERS}, {r: [] for r in MEMBERS}, {r: 0.0 for r in MEMBERS}
            clocks, idle_clocks = [], []
            for rep in range(REPS):
                win, pg, t_make, t_print, e, cl = self.rnd(input_seed(nonce, f"{name}:{seeds[i % len(seeds)]}", i, rep), ["anchor"] + order, h)
                makes.append(t_make)
                printing.append(t_print)
                clocks += cl
                for r in MEMBERS:
                    best[r].append(win[r])
                    ping[r].append(pg[r])
                    pings.append(pg[r])
                    errs[r] = max(errs[r], e[r])
                c = self.sample()
                if c is not None:
                    idle_clocks.append(c)
            read = "nvml_under_load" if clocks else "nvidia_smi_after_round" if idle_clocks else None
            clocks = clocks or idle_clocks
            row = {"index": i, "seed": seeds[i % len(seeds)], "input_sha256": h.hexdigest(), "order": order,
                   "anchor_position": "first", "clock_sm_mhz": min(clocks) if clocks else None,
                   "clock_read": read,
                   "windows_ms": {r: [round(v * 1000, 4) for v in best[r]] for r in MEMBERS},
                   "ping_ms": {r: [round(v * 1000, 4) for v in ping[r]] for r in MEMBERS},
                   "max_rel_err": {r: errs[r] for r in MEMBERS},
                   "bench_make_ms": [round(v * 1000, 4) for v in makes[-REPS:]],
                   "bench_print_ms": [round(v * 1000, 4) for v in printing[-REPS:]]}
            pairs.append(row)
        overhead = min(makes) + min(printing) + min(pings)
        self.partial = {"phase": name, "bench_overhead_ms": round(overhead * 1000, 4), "pairs": pairs}
        for row in pairs:
            for r in MEMBERS:
                v = (min(row["windows_ms"][r]) / 1000 - overhead) / SETS * 1000
                if not (math.isfinite(v) and v > 0):
                    raise Infra(f"{name} pair {row['index']}: {r} time {v:.4f} ms is not positive; the bench overhead exceeds the window")
                row[r + "_ms"] = round(v, 5)
        if len({p["input_sha256"] for p in pairs}) != len(pairs):
            raise Infra(f"{name}: two pairs got the same inputs")
        self.partial = None
        return {"run_id": f"{name}-{uuid.uuid4().hex}", "timer": TIMER, "measurement_mode": MODE,
                "accelerator_lease_id": lease, "gpu_uuid": gpu_uuid,
                "correctness": {"incumbent": True, "candidate": True, "anchor": True},
                "measurement_valid": True, "reps_per_pair": REPS, "bench_overhead_ms": round(overhead * 1000, 4), "pairs": pairs}

    def close(self):
        """Returns the cleanup errors; one member's error must not leave the others running."""
        errs = []
        for m in self.members.values():
            try:
                m.stop()
            except Exception as e:
                errs.append(f"{m.role}: {type(e).__name__}: {e}"[:200])
        if self.stage:
            shutil.rmtree(self.stage, ignore_errors=True)
        return errs


def check_request(req, rt):
    """Raises Infra for a request this backend cannot measure as asked."""
    need = lambda c, why: None if c else (_ for _ in ()).throw(Infra("bad request: " + why))
    need(req.get("schema_version") == VERSION, "schema_version")
    need(req.get("task") in tasks.TASKS, f"unknown task {req.get('task')!r}")
    need(req.get("timer") == TIMER, f"timer {req.get('timer')!r}, this backend is {TIMER}")
    need(req.get("measurement_mode") == MODE, "measurement_mode")
    need(req.get("runtime_sha256") == digest(rt), "runtime_sha256 is not this runtime")
    need(req.get("gpu_uuid") == rt["gpu_uuid"], "gpu_uuid is not this GPU")
    need(req.get("oracle_sha256") == sha(HERE / "tasks.py"), "oracle_sha256 is not this tasks.py")
    b = req.get("budget") or {}
    need(req.get("budget_sha256") == digest(b), "budget_sha256")
    for k in ("interaction_trials", "validation_trials"):
        need(type(b.get(k)) is int and 1 <= b[k] <= 50, k)
    need(type(b.get("wall_seconds")) in (int, float) and 0 < b["wall_seconds"] <= 600, "wall_seconds must be in (0, 600]")
    need(type(b.get("peak_vram_gb")) in (int, float) and 0 < b["peak_vram_gb"] <= 5, "peak_vram_gb must be in (0, 5]")
    s1, s2 = req.get("interaction_seeds"), req.get("validation_seeds")
    for s in (s1, s2):
        need(isinstance(s, list) and 2 <= len(s) <= 50 and all(type(x) is int for x in s) and len(set(s)) == len(s),
             "2 to 50 distinct integer seeds per phase")
    need(not set(s1) & set(s2), "interaction and validation seeds overlap")
    for role in MEMBERS:
        src = req.get(role) or {}
        need(isinstance(src.get("path"), str) and Path(src["path"]).is_file() and not Path(src["path"]).is_symlink(), role + " path")
        need(sha(src["path"]) == src.get("sha256"), role + " sha256 does not match the file")


def run(req, lease, job_id, timeout, override=None):
    """override maps role -> local path; the request (and its digest) keeps its own paths."""
    rt = runtime()
    digest_req = digest(req)
    req = json.loads(json.dumps(req))
    for role, path in (override or {}).items():
        if isinstance(req.get(role), dict):
            req[role]["path"] = path
    res = {"schema_version": VERSION, "request_sha256": digest_req, "task": req.get("task"), "timer": req.get("timer"),
           "runtime_sha256": req.get("runtime_sha256"), "gpu_uuid": req.get("gpu_uuid"),
           "oracle_sha256": req.get("oracle_sha256"), "budget_sha256": req.get("budget_sha256"),
           "sources": {r: (req.get(r) or {}).get("sha256") for r in MEMBERS}, "synthetic": False, "job_id": job_id,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "pending"}
    t0 = time.monotonic()
    bench = None
    try:
        check_request(req, rt)
        for role in ("incumbent", "candidate"):
            hit = re.search(ESCAPE, Path(req[role]["path"]).read_text())
            if hit:
                raise Fault(role, "import", f"source uses '{hit.group(0)}'; kernels may not start processes")
        bench = Bench(req, t0 + req["budget"]["wall_seconds"], timeout)
        nonce = os.urandom(16).hex()
        res["imports_s"] = bench.setup(req)
        res["isolation"] = bench.isolation
        res["warmup_ms"] = bench.warmup(nonce)
        res["compile_warmup_s"] = round(time.monotonic() - t0, 3)
        for name in ("interaction", "validation"):
            res[name] = bench.phase(name, req[name + "_seeds"], req["budget"][name + "_trials"], nonce, lease, rt["gpu_uuid"])
        if any(sha(req[r]["path"]) != req[r]["sha256"] for r in MEMBERS):
            raise Infra("a source file changed during the run")
        if bench.vram_peak is not None and (bench.vram_peak - bench.vram0) / 1024 > req["budget"]["peak_vram_gb"]:
            raise Infra(f"VRAM budget: {(bench.vram_peak - bench.vram0) / 1024:.2f} GB used")
        res["input_nonce"] = nonce
        res["status"] = "ok"
    except Fault as e:
        if e.role == "candidate":
            res.update(status="invalid", failure={"blame": "candidate", "stage": e.stage, "reason": e.reason})
        else:
            res.update(status="infrastructure", reason=f"{e.role} failed ({e.stage}): {e.reason}")
        res.pop("interaction", None), res.pop("validation", None)
    except Infra as e:
        res.update(status="infrastructure", reason=str(e))
        res.pop("interaction", None), res.pop("validation", None)
        if bench and bench.partial:
            res["diagnostic_rows"] = bench.partial  # not a measurement: the phase failed
    except Exception as e:  # a bench error is never the candidate's zero
        res.update(status="infrastructure", reason=f"bench error: {type(e).__name__}: {e}"[:400])
        res.pop("interaction", None), res.pop("validation", None)
    finally:
        if bench:
            errs = bench.close()
            if errs:
                res["cleanup_errors"] = errs
                if res["status"] == "ok":
                    res.update(status="infrastructure", reason="cleanup failed: " + "; ".join(errs))
                    res.pop("interaction", None), res.pop("validation", None), res.pop("input_nonce", None)
    res["wall_seconds"] = round(time.monotonic() - t0, 3)
    if bench and bench.vram_peak is not None:
        res["peak_vram_gb"] = round((bench.vram_peak - bench.vram0) / 1024, 3)
    res["runtime"] = rt
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--describe", action="store_true")
    a.add_argument("--request")
    a.add_argument("--output")
    a.add_argument("--lease-id")
    a.add_argument("--lock", help="take this flock for the run (when the caller does not hold the lease)")
    a.add_argument("--lock-wait", type=float, default=0)
    a.add_argument("--job-id")
    a.add_argument("--source", action="append", default=[], help="role=path: measure this local file (sha must match)")
    a.add_argument("--runtime-json", help="--describe output of the measuring host, for --build-request")
    a.add_argument("--timeout", type=float, default=60, help="seconds a member may take per window")
    a.add_argument("--build-request")
    a.add_argument("--task")
    a.add_argument("--incumbent")
    a.add_argument("--candidate")
    a.add_argument("--anchor")
    a.add_argument("--anchor-kind", choices=["library_op", "eager_reference"])
    a.add_argument("--usage", default="evaluation_only")
    a.add_argument("--round-id", default="adhoc")
    a.add_argument("--interaction-seeds", type=int, nargs="+")
    a.add_argument("--validation-seeds", type=int, nargs="+")
    a.add_argument("--interaction-trials", type=int, default=5)
    a.add_argument("--validation-trials", type=int, default=15)
    a.add_argument("--wall-seconds", type=float, default=600)
    a.add_argument("--peak-vram-gb", type=float, default=5)
    a = a.parse_args()
    if a.describe:
        rt = runtime()
        print(json.dumps({"runtime": rt, "runtime_sha256": digest(rt)}))
        return
    if a.build_request:
        rt = json.loads(Path(a.runtime_json).read_text())["runtime"] if a.runtime_json else None
        req = build_request(a, rt)
        Path(a.build_request).write_text(json.dumps(req, indent=1))
        print(json.dumps({"request": a.build_request, "request_sha256": digest(req)}))
        return
    req = json.loads(Path(a.request).read_text())
    job_id = a.job_id or "pair3-" + digest(req)[:16]
    lk = None
    lease = a.lease_id
    if a.lock:
        lk = open(a.lock, "a")
        t = time.monotonic()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t > a.lock_wait:
                    res = {"schema_version": VERSION, "request_sha256": digest(req), "job_id": job_id, "status": "pending",
                           "reason": "accelerator lock busy", "synthetic": False,
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    print(json.dumps(res))
                    sys.exit(3)
                time.sleep(1)
        lease = lease or f"flock:{a.lock}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
    override = dict(s.split("=", 1) for s in a.source)
    if set(override) - set(MEMBERS):
        sys.exit(f"--source roles must be in {MEMBERS}")
    def term(*_):
        raise Infra("terminated by the operator")
    signal.signal(signal.SIGTERM, term)
    res = run(req, lease or f"local:{uuid.uuid4().hex}", job_id, a.timeout, override)
    if a.output:
        tmp = Path(a.output).with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, a.output)
    print(json.dumps(res))
    sys.exit({"ok": 0, "invalid": 2}.get(res["status"], 3))


if __name__ == "__main__":
    main()
