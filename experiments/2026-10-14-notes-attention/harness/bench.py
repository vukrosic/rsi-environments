"""Score a kernel: time it against the reference on fresh inputs, alternating, and check every output.
  python bench.py attention start/attention.py [--seed 1234] [--trials 15]
Prints the speedup (reference time / kernel time); the last line is JSON. A wrong answer scores 0."""
import argparse, importlib.util, json, statistics, sys, time
import torch
import tasks

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("path")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--trials", type=int, default=15)
a = p.parse_args()
ref, make, tol = tasks.TASKS[a.name]
spec = importlib.util.spec_from_file_location("solution", a.path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kernel = mod.kernel
torch.manual_seed(a.seed)


def fail(msg):
    print(f"{a.name}: WRONG, {msg} -> score 0")
    print(json.dumps({"ok": False, "why": msg}))
    sys.exit(1)


def timed(f, sets):
    torch.mps.synchronize()
    t = time.perf_counter()
    outs = [f(*s) for s in sets]
    torch.mps.synchronize()
    return (time.perf_counter() - t) / len(sets) * 1000, outs


worst = 0.0
r, k = [], []
with torch.no_grad():
    for trial in range(3 + a.trials):  # 3 warm-up trials; every trial gets 4 new input sets, so nothing can be reused
        sets = [make() for _ in range(4)]
        if trial % 2:
            tk, got = timed(kernel, sets)
            tr, want = timed(ref, sets)
        else:
            tr, want = timed(ref, sets)
            tk, got = timed(kernel, sets)
        for g, w in zip(got, want):
            if not torch.is_tensor(g) or g.shape != w.shape:
                fail(f"output shape {getattr(g, 'shape', type(g))}, expected {tuple(w.shape)}")
            err = (g.float() - w).abs().max().item() / max(w.abs().max().item(), 1e-6)
            if not err <= tol:
                fail(f"relative error {err:.2e} in trial {trial}, allowed {tol:.0e}")
            worst = max(worst, err)
        if trial >= 3:
            r.append(tr)
            k.append(tk)
rm, km = statistics.median(r), statistics.median(k)
print(f"{a.name}: correct (relative error {worst:.1e}). reference {rm:.2f} ms, kernel {km:.2f} ms, speedup {rm / km:.2f}x")
print(json.dumps({"ok": True, "ref_ms": round(rm, 3), "kernel_ms": round(km, 3), "speedup": round(rm / km, 3), "err": worst}))
