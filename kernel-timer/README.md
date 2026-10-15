# Kernel timer (bench3)

A reward for "make this GPU kernel faster" that a kernel cannot fake by cheating the timer.

An LLM writes `kernel(...)`. `bench3.py` times it against PyTorch's own op for the same math (SDPA for attention,
`F.rms_norm` for rmsnorm, ...) and returns the speedup: baseline time / kernel time. Outputs are checked against a
plain PyTorch reference; a wrong answer scores 0. The kernel runs in its own process and, on Linux as root, under a
uid of its own, so it cannot read the reference, swap the clock or keep working after the clock stops.

```bash
python bench3.py attention start/attention.py --trials 15   # last line is JSON: ok, speedup, why, stage, fault, isolation, baseline, timer
python bench3.py attention my.py --baseline reference         # time against the plain reference instead
python hacks/run.py attention --bench bench3.py              # the 17 known fakes; exit 1 if any scores a gain
sudo sh remote/bench3_check.sh                               # full check on one CUDA GPU, ~5 min; exit 0 = pass
```

Needs Python 3.11+, torch, numpy. `tasks.py` holds 12 tasks: rmsnorm, softmax, swiglu, cross_entropy, attention,
attention_bwd, newton_schulz, selective_scan, sliding_window_attention, decode_attention, gqa_attention,
int4_matmul. 9 have a library baseline (`lib_<task>` in `tasks.py`); newton_schulz, selective_scan and int4_matmul
have no fused PyTorch op and are timed against the reference. `start/` has an unchanged kernel for 6 tasks: each
calls the baseline op, so it should read 1.0×.

For a league or king-of-the-hill reward (candidate vs the current best kernel, not vs PyTorch), see
[PAIRED.md](PAIRED.md): `pair3.py`, with a frozen anchor per task in `anchors/` and its own fakes in `hacks/pair/`.

## Why a library baseline

Timed against the plain reference, a kernel that only calls `F.softmax` read 3.0×. That rewards knowing the
library, not writing a faster kernel. Against the library op, the same kernel reads 1.0×.

## How one trial works

1. The bench draws a seed from `os.urandom`, builds 4 input sets, computes the reference answers (untimed) and a
   fingerprint of each: sum(answer × r), with r random.
2. It wakes the frozen kernel process and sends the seed; the clock starts. The kernel process rebuilds the same
   inputs, runs `kernel()` and sends back 4 fingerprints. The clock stops when they arrive and the process is frozen
   (SIGSTOP) at once.
3. The baseline runs in a second process of the same kind through the same window. The two alternate going first.
   The bench's own time (making inputs, fingerprints) and one ping floor, the fastest of the two sides' empty round
   trips, are subtracted from both windows, and each part is scored by its fastest trial. A non-positive time is an
   infrastructure failure (no verdict), never clamped.
4. Untimed: the outputs are copied out through shared memory and read twice, 30 ms apart. Every output must match
   the reference within tolerance, and its fingerprint must match the one sent inside the clock.

A fingerprint needs every element of the output, and the inputs are new each trial, so work cannot move outside the
clock. Each verdict says where it failed (`stage`: source, setup or run) and whose fault it was (`fault`: candidate
or infrastructure), so an RL trainer can drop infrastructure failures instead of punishing the policy for them.

Timing the baseline in the bench's own process read unchanged attention at 0.75×: the same SDPA call ran 9.2 ms
per 4 calls in the parent and 11.6 ms in a child. A second child process with alternating order removed that bias.

## Result: RTX A2000, 2026-10-15

torch 2.11.0+cu128, run as root, every row under the uid jail (`isolation: uid`), library baseline.

**Unchanged kernels** (should read 1.0×; 15 trials, two separate runs):

| Task | Baseline | Run 1 | Run 2 |
|---|---|---|---|
| swiglu | `F.silu(a) * b` | 0.994 | 0.986 |
| softmax | `torch.softmax` | 0.991 | 0.973 |
| rmsnorm | `F.rms_norm` | 0.976 | 0.998 |
| cross_entropy | `F.cross_entropy` | 0.973 | 0.976 |
| newton_schulz | reference | 0.984 | 0.990 |
| attention | SDPA, causal | 0.995 | 0.997 |

All 12 rows read 0.973–0.998×.

**16 known fakes** on attention (`hacks/attention/`, one file each, the docstring says what it tries): 0 of 16
scored a gain, with the source scan on and off. A 17th, `lateping`, was found later; see "Ping exploit" below.

| Result | Fakes |
|---|---|
| Score 0, wrong values | alias, aten, cache, lazy, mutate, replay, wrongref, zeros |
| Score 0, crashed | half (returns float16), peek (the bench's globals do not exist in the kernel's process) |
| No gain, 0.96–1.01× | clock, escape, fallback, slowref, thread |
| No gain, 0.054× | trace |

`escape` is the strongest: it starts a process in a new session that burns the GPU whenever the kernel is frozen, so
the baseline would be timed on a busy GPU. Under the uid jail the bench freezes every process of that uid, so it
scored 0.96–1.01×. `remote/jail_test.py` checks this on the CPU: a setsid grandchild is frozen, woken and killed,
and a process of another uid is never signalled (10/10 checks passed).

## Ping exploit (fixed in timer v3)

Timer v2 subtracted each side's own empty round trip. A kernel owns its process's stdin, so it can answer the ping
late while the timed work stays the same, and every ms of delay came off its time:

| Kernel | v2 | v3 |
|---|---|---|
| rmsnorm, ping 1 ms late | 2.90× | 0.968× |
| rmsnorm, ping 3 ms late | 1514× (time clamped to ~0) | 0.974× |
| attention, ping 3 ms late (`hacks/attention/lateping.py`) | 1.36× | 0.999× |

v3 subtracts one floor from both sides, so a kernel can raise its own ping but not lower the floor. v2 also clamped a
non-positive time to 1e-6 s; a trainer that clips the reward would have paid such a kernel the maximum. The 16
earlier fakes still score no gain on v3 with the source scan off. Every verdict carries `"timer": "v3"`.

## A first model trained on this reward

Qwen2.5-Coder-1.5B-Instruct with a LoRA adapter (2.2 M weights), 36 GRPO steps of 8 samples on 9 tasks (timer v2),
then 16 samples per task on all 12 (3 held out):

| | Base | After 36 steps |
|---|---|---|
| Correct, 9 training tasks | 12/144 | 13/144 |
| Correct, 3 held-out tasks | 5/48 | 2/48 |

No measured gain (Fisher p = 1.0 and 0.44). 29 of 36 steps had no correct sample, so GRPO updated the model 7 times.

The largest speedups were compiler wrappers, not kernels: `@torch.jit.script` on the swiglu formula (1.64×, a real
fusion against `F.silu(a) * b`), `@torch.compile` on the newton_schulz reference (1.12×), and the selective_scan
reference copied unchanged under `@torch.jit.script` (4.7×, in a repeat with another sampling seed). For the three
tasks timed against an eager reference, a compiler wrapper is the cheapest way to earn reward. A baseline of the
compiled reference would close that.

## Limits

- 16 fakes scoring 0 does not prove no fake exists. New fakes are welcome as issues.
- Without root (or on macOS) there is no uid jail (`isolation: pgroup`); `escape` is then not blocked by design.
- One GPU model tested. Unchanged kernels read 0.97–1.0×, so a 1.15× threshold is used for "gain".
- One small model, one short run, one seed per eval so far; see above.
