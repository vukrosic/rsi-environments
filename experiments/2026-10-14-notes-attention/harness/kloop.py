"""Kernel batch: each agent works in its own sandboxed folder on agent/kernel.py and may time it 10 times. One
server times one kernel at a time, so agents never share the GPU during a measurement.
  python kloop.py start --ws W plain-1 plain-2 critic-1 critic-2   calibrate, make the folders, launch the agents
  python kloop.py tick --ws W --for 540                             time queued kernels one at a time, then the finals
  python kloop.py collect --ws W --out kbatch1                      copy each agent's work and a readable trace
  python kloop.py notes --ws W --out N.md lessons-r1-1 ...          join the agents' LESSONS.md for the next round
  python kloop.py start --ws W --notes N.md lessons-r2-1 ...        lessons-* agents get N.md as LESSONS.md
  python kloop.py start --ws W --kernel cross_entropy --notes N.md transfer-1 control-1   another kernel (agent-<kernel>/)
  advice-* agents get advice-generic.md as LESSONS.md: general advice as long as the notes, from no agent's runs
tick and collect read the kernel from W/base.json."""
import argparse, json, os, re, shutil, statistics, subprocess, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
rsi = here.parent
sys.path.insert(0, str(rsi / "loop"))
import loop  # sandbox profile, alive(), flags()

PY, RUNS, SEED, FINAL_SEED, BUSY = loop.PY, 10, 1234, 777, 1.10

BASE = "Make kernel() as fast as you can."
PROMPTS = {
    "plain": BASE,
    "critic": BASE + " Before your first bench run, write 5 different ideas. For each, write the strongest reason it could"
                     " fail to make the kernel faster, then rank the 5. Before each later run, redo the ranking with the"
                     " times you have seen.",
    "control": BASE,
    "lessons": BASE + " LESSONS.md in this folder holds notes from earlier agents who did this same task. Read it before"
                      " your first bench run. Before you finish, rewrite LESSONS.md for the next agents, who start from the"
                      " unchanged kernel.py: what made the kernel faster, what did not, and the times you measured. Keep what"
                      " is still true from the notes you were given. At most 300 words, in words, no code.",
}
PROMPTS["transfer"] = BASE + (" LESSONS.md in this folder holds notes from earlier agents who optimized a different kernel"
                              " (causal attention) on this same Mac. Read it before your first bench run.")
PROMPTS["advice"] = BASE + (" LESSONS.md in this folder holds general advice on optimizing GPU kernels on this Mac."
                            " Read it before your first bench run.")
PROMPTS["notes"] = BASE + " LESSONS.md in this folder holds notes from earlier agents. Read it before your first bench run."
ADVICE = here / "advice-generic.md"  # a control for the notes: as long, but not written from any agent's runs
NO_NOTES = "No notes yet: you are the first agents on this task.\n"

TASK = """This folder holds one GPU kernel task on an Apple M4 Mac (PyTorch {torch} on the GPU, device "mps").

- kernel.py: {sig}, {what}. It is the only code file you may edit.
- reference.py: the reference answer and the inputs it is timed on ({inputs}).
- bench.py: times kernel.py against reference() on fresh inputs and prints the speedup. You may use it {runs} times: {py} bench.py
  A run usually returns in under a minute; give the command a timeout of at least 10 minutes. If it is cut off, {py} bench.py --wait gets the result.
- check.py: checks kernel.py's answer without timing. Use it as often as you like: {py} check.py

Rules:
- {sig} keeps its name and arguments and returns a GPU tensor with the reference's shape, within the tolerance in reference.py.
- Anything that runs on this Mac counts, for example a custom Metal shader through torch.mps.compile_shader.
- kernel.py may not read any file.
- Time only with bench.py; other jobs share this GPU.
- Work only inside this folder. Do not read, list or search anything outside it.

Unchanged, kernel.py runs in {ms:.1f} ms (speedup 1.00x).

{prompt}

Your final kernel.py is timed again at the end on new inputs. When you are done, run check.py. End your reply with one line: EDIT: <one sentence saying what your final kernel.py does>"""


KERNEL = "attention"
KERNELS = {  # name: (signature, what it computes, the timed inputs)
    "attention": ("kernel(q, k, v)", "causal attention", "batch 8, 8 heads, 1024 positions, head size 64, float32"),
    "cross_entropy": ("kernel(logits, y)", "the mean cross-entropy loss",
                      "logits of 8192 rows by 4096 classes, float32, and 8192 integer labels"),
    "attention_bwd": ("kernel(q, k, v, do)", "the gradients of causal attention (dq, dk, dv stacked into one tensor)",
                      "batch 8, 8 heads, 1024 positions, head size 64, float32; do is the gradient of the output"),
    "newton_schulz": ("kernel(g)", "Muon's Newton-Schulz orthogonalization (5 iterations)",
                      "a stack of 10 matrices of 704 by 256, float32"),
    "selective_scan": ("kernel(x, dt, A, Bm, C)", "Mamba's selective scan over 128 time steps",
                       "x and dt of batch 8, 128 steps, 256 channels; A of 256 by 16 states; Bm and C of batch 8, 128 steps, 16 states; float32"),
    "sliding_window_attention": ("kernel(q, k, v)", "causal attention where each position sees only the last 128 positions",
                                 "batch 1, 8 heads, 2048 positions, head size 64, float32"),
    "decode_attention": ("kernel(q, k, v)", "attention of one new token over a cache of 4096 keys and values",
                         "q of batch 4, 8 heads, 1 position, head size 128; k and v with 4096 positions; float32"),
    "gqa_attention": ("kernel(q, k, v)", "causal grouped-query attention (32 query heads, 8 key/value heads)",
                      "q of batch 1, 32 heads, 1024 positions, head size 64; k and v with 8 heads; float32"),
    "int4_matmul": ("kernel(x, w, scale)", "x times a 4-bit weight matrix (two weights per byte, one scale per 128 weights)",
                    "x of 32 by 4096 float16, w of 4096 by 2048 uint8, scale of 4096 by 32 float16"),
}


def agent_dir():
    return here / ("agent" if KERNEL == "attention" else f"agent-{KERNEL}")


def use(ws):
    global KERNEL
    KERNEL = json.loads((ws / "base.json").read_text()).get("kernel", "attention")


HEAVY = {"LeagueofLegends": "a game", "chrome-headless-shell": "a Remotion render"}  # slowed the GPU 15-40%, 2026-10-14


def heavy():
    """What else is using the GPU, or None."""
    for name, what in HEAVY.items():
        if subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0:
            return f"{what} ({name}) is running"


def bench(path, seed=SEED):
    out = subprocess.run([PY, "-W", "ignore", str(here / "bench.py"), KERNEL, str(path), "--seed", str(seed)],
                         cwd=here, capture_output=True, text=True, timeout=600)
    lines = (out.stdout + out.stderr).strip().splitlines()
    try:
        r = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        r = {"ok": False, "why": "crashed"}
    shown = lines[0] if r.get("why") != "crashed" else "kernel.py crashed:\n" + "\n".join(lines[-12:])
    return r, shown


def measured(path, base, seed=SEED):
    """Bench; if the reference ran more than 10% slower than on the idle GPU, something else used it: measure again."""
    tries = []
    for _ in range(3):
        r, shown = bench(path, seed)
        tries.append((r.get("ref_ms", 0), r, shown))
        if not r.get("ok") or r["ref_ms"] <= BUSY * base:
            break
    _, r, shown = min(tries, key=lambda t: t[0] if t[1].get("ok") else 0)
    r["tries"] = len(tries)
    if r.get("ok") and r["ref_ms"] > BUSY * base:
        r["busy"] = True
        shown += " (the GPU was busy during this run)"
    return r, shown


def start(ws, ids, notes=None):
    if heavy() and not os.environ.get("KLOOP_BUSY_OK"):
        sys.exit("GPU not idle: " + heavy() + ". Set KLOOP_BUSY_OK=1 to start anyway.")
    ms = [bench(agent_dir() / "kernel.py")[0]["ref_ms"] for _ in range(3)]
    base = {"ref_ms": statistics.median(ms), "kernel": KERNEL}
    (ws / "base.json").write_text(json.dumps(base))
    print("idle reference", ms)
    import torch
    top = (Path.home() / ".codex/config.toml").read_text().split("\n[")[0]
    conf = [l for l in top.splitlines() if re.match(r"(model|model_reasoning_effort|service_tier) *=", l)]
    for i in ids:
        root = ws / i
        task, home, server = root / "task", root / "home", root / "server"
        for d in (task, home, server):
            d.mkdir(parents=True, exist_ok=True)
        for f in ("kernel.py", "reference.py", "check.py", "bench.py"):
            shutil.copy(agent_dir() / f, task / f)
        shutil.copy(agent_dir() / "kernel.py", server / "start.py")
        if i.startswith(("lessons", "transfer", "notes")):
            (task / "LESSONS.md").write_text(Path(notes).read_text() if notes else NO_NOTES)
            shutil.copy(task / "LESSONS.md", server / "LESSONS_given.md")
        if i.startswith("advice"):
            shutil.copy(ADVICE, task / "LESSONS.md")
            shutil.copy(ADVICE, server / "LESSONS_given.md")
        (home / "config.toml").write_text("\n".join(conf) + "\n\n[features]\nmemories = false\n")
        shutil.copy(Path.home() / ".codex/auth.json", home / "auth.json")
        os.chmod(home / "auth.json", 0o600)
        (root / "sandbox.sb").write_text(loop.SANDBOX.format(task=task, home=home))
        sig, what, inputs = KERNELS[KERNEL]
        prompt = TASK.format(sig=sig, what=what, inputs=inputs, torch=torch.__version__, runs=RUNS, py=PY, ms=base["ref_ms"], prompt=PROMPTS[i.split("-")[0]])
        (root / "prompt.txt").write_text(prompt)
        cmd = ["sandbox-exec", "-f", str(root / "sandbox.sb"), "codex", "exec", "-C", str(task), "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "-o", str(home / "last.txt"), prompt]
        pr = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=open(root / "log.txt", "w"), stderr=subprocess.STDOUT,
                              env={**os.environ, "CODEX_HOME": str(home)}, start_new_session=True)
        (root / "agent.json").write_text(json.dumps({"pid": pr.pid, "t0": time.time()}))
        print(i, "started, pid", pr.pid)


def results(root):
    p = root / "server/results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def tokens(root):
    lines = (root / "log.txt").read_text(errors="replace").splitlines()
    return sum(int(lines[n + 1].replace(",", "")) for n, l in enumerate(lines[:-1])
               if l == "tokens used" and lines[n + 1].replace(",", "").isdigit())


def tick(ws, ids, secs):
    base = json.loads((ws / "base.json").read_text())["ref_ms"]
    end = time.time() + secs
    while True:
        waiting = []
        for i in ids:
            root = ws / i
            done = {r["job"] for r in results(root)}
            q = root / "task/queue"
            for f in sorted(q.glob("run*.py")) if q.exists() else []:
                if f.stem not in done and int(f.stem[3:]) <= RUNS:
                    waiting.append((f.stat().st_mtime, i, f.stem, f))
            a = json.loads((root / "agent.json").read_text())
            if not loop.alive(a["pid"]) and "final" not in done and not any(w[1] == i for w in waiting):
                a.setdefault("t_end", time.time())
                (root / "agent.json").write_text(json.dumps(a))
                waiting.append((float("inf"), i, "final", root / "task/kernel.py"))
        if waiting:
            t_req, i, name, src = min(waiting, key=lambda w: w[0])
            root = ws / i
            copy = root / "server" / f"{name}.py"
            shutil.copy(src, copy)  # time a frozen copy the agent cannot change mid-run
            t0 = time.time()
            if name == "final":  # 5 timings in a row on new inputs, median
                rs = [measured(copy, base, FINAL_SEED + n)[0] for n in range(5)]
                ok = [r for r in rs if r.get("ok")]
                r = {"ok": len(ok) == 5, "kernel_ms": statistics.median([x["kernel_ms"] for x in ok]) if ok else None,
                     "ref_ms": statistics.median([x["ref_ms"] for x in ok]) if ok else None, "all": rs}
            else:
                r, shown = measured(copy, base)
                tmp = root / "task/results" / f".{name}.tmp"
                tmp.parent.mkdir(exist_ok=True)
                tmp.write_text(shown + "\n")
                os.rename(tmp, root / "task/results" / f"{name}.txt")
            r.update(job=name, waited_secs=round(t0 - (t0 if t_req == float("inf") else t_req)),
                     bench_secs=round(time.time() - t0))
            with open(root / "server/results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(time.strftime("%H:%M:%S"), i, name, r.get("kernel_ms"), "ms" if r.get("ok") else r.get("why", ""),
                  f"(waited {r['waited_secs']} s, {r.get('tries', 5)} timing)", flush=True)
            continue
        for i in ids:
            for fl in loop.flags(ws, i):
                print("FLAG", fl)
        all_done = all("final" in {r["job"] for r in results(ws / i)} for i in ids)
        if all_done or time.time() > end:
            break
        time.sleep(1)
    for i in ids:
        root = ws / i
        a = json.loads((root / "agent.json").read_text())
        state = "working" if loop.alive(a["pid"]) else "finished"
        print(f"{i}: {state}, {round((a.get('t_end', time.time()) - a['t0']) / 60)} min, {tokens(root):,} tokens |",
              ", ".join(f"{r['job']} {r.get('kernel_ms') or 'wrong'}" for r in results(root)) or "no runs yet")
    print("ALL DONE" if all_done else "not done")


def collect(ws, ids, out):
    """Copy each agent's kernels and write TRACE.md: prompt, runs table, then what it said, ran and changed."""
    import difflib, readable
    rows = []
    for i in ids:
        root, dst = ws / i, here / out / i
        dst.mkdir(parents=True, exist_ok=True)
        rs = results(root)
        a = json.loads((root / "agent.json").read_text())
        for f in sorted((root / "server").glob("*.py")):
            shutil.copy(f, dst / f.name)
        (dst / "log.txt").write_text((root / "log.txt").read_text(errors="replace").replace(str(ws), "<ws>"))
        for f in ("task/LESSONS.md", "server/LESSONS_given.md"):
            if (root / f).exists():
                shutil.copy(root / f, dst / Path(f).name.replace("LESSONS.md", "LESSONS_written.md"))
        (dst / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        prompt = (root / "prompt.txt").read_text().split("(speedup 1.00x).\n\n")[1].split("\n\n")[0]
        md = [f"# Agent `{i}`", "", f"**Prompt:** {prompt}", "",
              f"{round((a['t_end'] - a['t0']) / 60)} min, {tokens(root):,} tokens. Unchanged code: {json.loads((ws / "base.json").read_text())["ref_ms"]:.1f} ms.", "",
              "| Run | Kernel time | Speedup over the reference |", "|---|---|---|"]
        for r in rs:
            md.append(f"| {r['job']} | {str(r['kernel_ms']) + ' ms' if r.get('kernel_ms') else 'wrong: ' + r.get('why', '')} "
                      f"| {str(round(r['ref_ms'] / r['kernel_ms'], 2)) + 'x' if r.get('kernel_ms') else ''} |")
        md += ["", "`final` is the last kernel.py, timed 5 times on new inputs (median).", "", "## What it did, in order", ""]
        n = 0
        for kind, text in readable.events(root / "log.txt", ws, root):
            if kind == "say":
                md += ["> " + text.replace("\n", "\n> "), ""]
            elif kind == "run":
                n += 1
                prev = root / ("server/start.py" if n == 1 else f"server/run{n - 1}.py")
                cur = root / f"server/run{n}.py"
                md += [f"### Bench run {n}", ""]
                if cur.exists():
                    d = "".join(difflib.unified_diff(prev.read_text().splitlines(keepends=True),
                                                     cur.read_text().splitlines(keepends=True), "before", "after"))
                    md += [f"<details><summary>Code change for run {n}</summary>", "", "```diff", d, "```", "", "</details>", ""]
                md += [text, ""]
            else:
                md += [f"- {text}"]
        (dst / "TRACE.md").write_text("\n".join(md) + "\n")
        fin = next((r for r in rs if r["job"] == "final"), {})
        best = min((r["kernel_ms"] for r in rs if r.get("kernel_ms") and r["job"] != "final"), default=None)
        rows.append((i, fin.get("kernel_ms"), best, round((a["t_end"] - a["t0"]) / 60), tokens(root)))
        print(rows[-1])
    return rows


def notes(ws, ids, out):
    """Join the notes the agents wrote into the LESSONS.md the next round reads."""
    parts = ["Notes from the previous round of agents, one section per agent.", ""]
    for n, i in enumerate(ids, 1):
        text = (ws / i / "task/LESSONS.md").read_text().strip()
        print(i, len(text.split()), "words")
        parts += [f"## Agent {n}", "", text, ""]
    Path(out).write_text("\n".join(parts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["start", "tick", "collect", "notes"])
    p.add_argument("ids", nargs="*", default=["plain-1", "plain-2", "critic-1", "critic-2"])
    p.add_argument("--ws", required=True)
    p.add_argument("--for", dest="secs", type=int, default=540)
    p.add_argument("--out", default="kbatch1")
    p.add_argument("--notes", default=None)
    p.add_argument("--kernel", default="attention", choices=list(KERNELS))
    a = p.parse_args()
    ws = Path(a.ws).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    if a.cmd == "start":
        KERNEL = a.kernel
    elif a.cmd in ("tick", "collect"):
        use(ws)
    {"start": lambda: start(ws, a.ids, a.notes), "notes": lambda: notes(ws, a.ids, a.out), "tick": lambda: tick(ws, a.ids, a.secs),
     "collect": lambda: collect(ws, a.ids, a.out)}[a.cmd]()
