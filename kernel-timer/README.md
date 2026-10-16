# Kernel timer (bench3)

A GPU kernel benchmark with output checks, timing controls and separate candidate/infrastructure verdicts.

**Known reward-integrity gaps:** the published v5 source has predictable timed
inputs and an incomplete fingerprint-length check. Keep its speedups out of RL
updates and incumbent promotion pending repairs and validation. See
[the source-review advisory](KNOWN-LIMITATIONS.md).

An LLM writes `kernel(...)`. `bench3.py` times it against the fastest version of the same math that needs no kernel
work (PyTorch's own op, or the plain reference, eager or compiled) and returns the speedup: baseline time / kernel
time. Outputs are checked against a plain PyTorch reference; a wrong answer scores 0. The kernel runs in its own
process and, on Linux as root, under a uid of its own. These controls provide process separation; they do not
establish a security sandbox for arbitrary hostile code. See the [dated v5 checks and limits](VALIDATION-V5.json).

```bash
python bench3.py attention start/attention.py --trials 10   # last line is JSON: ok, speedup, why, stage, fault, isolation, baseline, timer
python bench3.py attention my.py --baseline library          # time against PyTorch's op instead (or --baseline reference)
python hacks/run.py attention --bench bench3.py             # the 20 known fakes; exit 1 if any scores a gain
python tools/fastest_baseline.py                            # measure the fastest baseline per task on your GPU
sudo sh remote/bench3_check.sh                              # full check on one CUDA GPU, ~25 min; exit 0 = pass
```

Needs Python 3.11+, torch, numpy. `tasks.py` holds 12 tasks: rmsnorm, softmax, swiglu, cross_entropy, attention,
attention_bwd, newton_schulz, selective_scan, sliding_window_attention, decode_attention, gqa_attention,
int4_matmul. 9 have a library op (`lib_<task>` in `tasks.py`). `start/` has an unchanged kernel for 6 tasks: each
calls the library op, so against `--baseline library` it should read 1.0×.

For a league or king-of-the-hill reward (candidate vs the current best kernel, not vs PyTorch), see
[PAIRED.md](PAIRED.md): `pair3.py`, with a frozen anchor per task in `anchors/` and its own fakes in `hacks/pair/`.

## Which baseline

Timed against the plain reference, a kernel that only calls `F.softmax` read 3.0×. That rewards knowing the library,
not writing a faster kernel, so timer v2 used the library op. Then a trained model's best "kernels" turned out to be
the reference under a compiler. Against the library op (RTX A2000, 10 trials):

| No-effort kernel | Speedup over the library op |
|---|---|
| selective_scan reference under `torch.compile` / `torch.jit.script` | 7.27× / 4.59× |
| int4_matmul reference under `torch.compile` | 3.18× |
| decode_attention: the eager reference (SDPA is slow at one query token) | 2.55× |
| gqa_attention reference under `torch.compile` | 1.95× |
| cross_entropy reference under `torch.compile` | 1.85× |
| swiglu: `F.silu(a) * b` under `torch.compile` | 1.61× |

Since timer v4 the default baseline is the fastest of {library op, eager reference, `torch.compile` of each},
measured per task on this GPU and stored in `tasks.FASTEST`. A compiled baseline is built once with AOTInductor
(ahead of time) and cached, because `torch.compile` at score time took over 6 minutes per process on
selective_scan. A one-line compiler wrapper now reads about 1.0×; the swiglu compile wrapper reads 1.56× against
`--baseline library` and 0.98× against the default. Another GPU or torch version should re-run
`tools/fastest_baseline.py`.

## How one trial works

1. The bench draws a seed from `os.urandom`, builds 4 input sets, computes the reference answers (untimed) and a
   fingerprint of each: sum(answer × r), with r random.
2. It wakes the frozen kernel process and sends the seed; the clock starts. The kernel process rebuilds the same
   inputs, runs `kernel()` and sends back 4 fingerprints. The clock stops when they arrive and the process is frozen
   (SIGSTOP) at once.
3. The baseline runs in a second process of the same kind through the same window; the two alternate going first.
   Each side's timed window comes right after an untimed window of its own on other inputs. After every window the
   bench times a small matmul until the GPU reads idle (step "drain", below). The baseline process's own time for
   making inputs and fingerprints, and one ping floor (the fastest empty round trip of either side), are subtracted
   from both windows. The score is the median, over pairs of trials (one with each side first), of summed baseline
   time / summed kernel time. A non-positive time is an infrastructure failure (no verdict), never clamped.
4. Untimed: the outputs are copied out through shared memory and read twice, 30 ms apart. Every output must match
   the reference within tolerance, and its fingerprint must match the one sent inside the clock.

A fingerprint needs every element of the output, and the inputs are new each trial, so work cannot move outside the
clock. Each verdict says where it failed (`stage`: source, setup or run) and whose fault it was (`fault`: candidate
or infrastructure), so an RL trainer can drop infrastructure failures instead of punishing the policy for them.

Timing the baseline in the bench's own process read unchanged attention at 0.75×: the same SDPA call ran 9.2 ms
per 4 calls in the parent and 11.6 ms in a child. A second child process with alternating order removed that bias.

## Timer v5: same kernel on both sides should read 1.0× (A/A)

RTX A2000, torch 2.11.0+cu128, uid jail, the eager reference timed against itself, 3 runs of 10 trials. v4 scored
each side by its fastest trial and subtracted the bench's own input-making time; v5 is the scheme above.

| Task | v4 | v5 |
|---|---|---|
| rmsnorm, softmax, swiglu, cross_entropy, attention, sliding_window, gqa, int4_matmul, attention_bwd | 0.984–1.004 | 0.989–1.008 |
| newton_schulz | 1.0–1.027 | 0.995–1.002 |
| decode_attention | 1.124–1.162 | 0.953–0.994 |
| selective_scan | 0.941–1.071 | 0.974–1.052 |

Three causes, each found with per-trial logs. The bench calls `torch.cuda.empty_cache()` between trials, so its own
input making swung 3.5–5.5 ms while each child's held 3.0–3.6 ms; v5 subtracts the baseline child's own. The first
window after an idle gap ran up to 50% slower (newton_schulz 15–21 vs 14 ms). And a process that just ran is 1–5%
faster: one untimed baseline window before each pair made every task lean to the baseline (rmsnorm 0.975–0.979).
Heating the GPU with 20 ms of matmuls before each trial made it worse (windows 18–27 ms instead of 14–21). An untimed
window by the same side right before its timed one fixed newton_schulz and decode.

selective_scan keeps a per-process offset: one process ran 58.8 ms and the other 62.0 ms for a whole run, in either
order. CPU pinning, a fixed hash seed and a random GPU allocation before each trial did not remove it. Use a wider
promotion margin for selective_scan (6%) and decode_attention (5%) than for the rest (2%).

**Real gains survive v5** (`--baseline library`, 2 runs each):

| Kernel | v3 | v5 |
|---|---|---|
| cross_entropy reference under `torch.compile` | 1.872, 1.838 | 1.706, 1.704 |
| swiglu under `torch.compile` | 1.537, 1.537 | 1.572, 1.565 |
| swiglu under `torch.jit.script` | 1.612, 1.625 | 1.654, 1.662 |
| decode_attention, a model-written kernel | 2.615, 2.429 | 2.074, 2.032 |

## Spill: work left on the GPU for the opponent (fixed in timer v5)

The kernel process syncs the GPU before it answers, but the bench fingerprints the outputs after that sync, and a
kernel can return a tensor subclass. Its `.float()`, called by the fingerprint, can queue GPU work on a side stream.
The clock stops and the process is frozen; the queued work is not. It runs during the baseline's next window:

| Fake (`hacks/attention/`) | What it queues | v5 without the drain | v5 |
|---|---|---|---|
| `spill.py` | 40 big matmuls on every `.float()` | 0.034× (landed in its own window) | 0.40× |
| `spill2.py` | 2 ms GPU sleep, then 20 matmuls, on every 4th `.float()` | 0.267× (hit its own timed window) | 0.85–0.87× |
| `spill3.py` | the same, only after its timed window | **1.62×** (baseline window 3.2 → 5.2 ms) | 0.84–0.92× |

`spill3` computes attention exactly like SDPA. The drain: after every window the bench times a 512² matmul until it
reads ≤ 3× its idle time + 0.2 ms; past 1 s the side that left the work is faulted (an infrastructure fault if it was
the baseline).

## Known fakes

20 fakes on attention, one file each in `hacks/attention/`; the docstring says what it tries. Under v5 with the
source scan off, 0 of 20 scored a gain (RTX A2000). Under v3 the first 16 also scored 0 with the scan
on.

| Result | Fakes |
|---|---|
| Score 0, wrong values | aten, cache, mutate, replay, wrongref, zeros |
| Score 0, output changed after the clock | lazy |
| Score 0, crashed | half (returns float16), peek (the bench's globals do not exist in the kernel's process) |
| No gain, 0.87–1.03× | alias, clock, escape, fallback, lateping, slowref, spill2, spill3, thread |
| No gain, under 0.4× | spill (0.398×), trace (0.053×) |

`escape` starts a process in a new session that burns the GPU whenever the kernel is frozen, so the baseline would
be timed on a busy GPU. Under the uid jail the bench freezes every process of that uid, so it scores no gain.
`remote/jail_test.py` checks this on the CPU: a setsid grandchild is frozen, woken and killed, and a process of
another uid is never signalled.

## Ping exploit (fixed in timer v3)

Timer v2 subtracted each side's own empty round trip. A kernel owns its process's stdin, so it can answer the ping
late while the timed work stays the same, and every ms of delay came off its time:

| Kernel | v2 | v3 |
|---|---|---|
| rmsnorm, ping 1 ms late | 2.90× | 0.968× |
| rmsnorm, ping 3 ms late | 1514× (time clamped to ~0) | 0.974× |
| attention, ping 3 ms late (`hacks/attention/lateping.py`) | 1.36× | 0.999× |

Since v3 one floor comes off both sides, so a kernel can raise its own ping but not lower the floor. v2 also clamped
a non-positive time to 1e-6 s; a trainer that clips the reward would have paid such a kernel the maximum. Every
verdict carries its timer version (`"timer": "v5"`).

## A first model trained on this reward

Qwen2.5-Coder-1.5B-Instruct with a LoRA adapter (2.2 M weights), 36 GRPO steps of 8 samples on 9 tasks (timer v2),
then 16 samples per task on all 12 (3 held out), two sampling seeds:

| | Base | After 36 steps |
|---|---|---|
| Correct, 9 training tasks | 26/288 | 21/288 |
| Correct, 3 held-out tasks | 8/96 | 11/96 |

No measured gain; the two seeds disagree in sign. 29 of 36 steps had no correct sample, so GRPO updated the model 7
times. The largest speedups were compiler wrappers (swiglu under `torch.jit.script` 1.64×, the selective_scan
reference under `torch.jit.script` 4.7×), which led to the fastest-baseline change above.

## Limits

- 20 fakes scoring no gain does not prove no fake exists. New fakes are welcome as issues.
- Without root (or on macOS) there is no uid jail (`isolation: pgroup`); `escape` is then not blocked by design.
- One GPU model tested (RTX A2000). `tasks.FASTEST` was measured on it; on another GPU re-run
  `tools/fastest_baseline.py`.
- selective_scan and decode_attention A/A spread ±5%; see the margins above.
- One small model, one short run so far.
