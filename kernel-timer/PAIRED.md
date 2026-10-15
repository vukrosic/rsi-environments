# Paired timer (pair3): candidate vs incumbent

`bench3.py` scores one kernel against the library op. A league or king-of-the-hill reward needs something else:
is the **candidate** faster than the **current best kernel** (the incumbent), measured in the same job, on the same
inputs, in an order that neither side controls? `pair3.py` answers that and writes one JSON receipt per job.

```bash
python pair3.py --describe > runtime.json                     # this machine's runtime identity (GPU uuid, torch, file hashes)
python pair3.py --runtime-json runtime.json --build-request req.json --task rmsnorm \
    --incumbent start/rmsnorm.py --candidate my.py \
    --interaction-seeds 1 2 3 4 5 --validation-seeds 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
python pair3.py --request req.json --output receipt.json --lock /tmp/gpu.lock --lock-wait 600
python tools/aa_report.py receipt.json                        # medians by order, anchor ratios, clocks
```

Exit 0 = `ok`, 2 = `invalid` (the candidate failed a check), 3 = `infrastructure` (no verdict; retry, never score
it as 0). Needs the same Python/torch/numpy as `bench3.py`; the uid jail needs Linux and root.

## How a job works

1. Three members, each in its own process under its own random uid: incumbent, candidate and a frozen **anchor**
   (`anchors/<task>.py`, the library op), which is timed in every pair and reported only. If incumbent and
   candidate drift but the anchor drifts with them, the GPU changed, not the kernels.
2. Warm-up: 4 untimed rounds. The competitors swap order each round and the anchor always goes last (finding 4).
3. Each pair runs 3 rounds on fresh inputs. In a round the anchor goes first, then incumbent and candidate, which
   swap order every pair. Before every window the bench runs 20 ms of its own GPU load (finding 1).
4. A member's pair time = (its fastest window − the bench's fastest input-making and fingerprint time − the fastest
   empty round trip of **any** member in the phase) / 4 input sets. A non-positive time is an infrastructure failure,
   never clamped (finding 3).
5. Two phases: interaction (what a learner may see) and validation (held out, different seeds). The inputs come
   from a nonce drawn at run time and published in the receipt afterwards, so every input can be rebuilt.
6. Every output is checked against the reference, and its fingerprint must match the one sent inside the clock
   (as in `bench3.py`). A wrong candidate is `invalid`, blame candidate; a wrong incumbent or anchor is
   `infrastructure`.
7. After every time a member is woken (import, each window, each output copy) the GPU must be idle again within
   1 s. Work a frozen process left queued would slow whoever runs next; this check blames the member that queued it.
8. Each pair records the SM clock, read through NVML at the end of the preheat while the GPU is loaded.

## What we found while building it (RTX A2000 6 GB, torch 2.11, 2026-10-15)

Ratios below are the median over pairs of candidate time / incumbent time. In an A/A run (same file on both sides)
it should be 1.00.

### 1. The second member of a round ran faster

| Timer | Incumbent first | Candidate first | Median reported |
|---|---|---|---|
| pair3-v2, run 1 | 0.959 | 1.036 | 0.963 |
| pair3-v2, run 2 | 0.970 | 1.020 | 0.979 |
| pair3-v3 (20 ms GPU preheat before every window), run 1 | 0.990 | 0.999 | 0.995 |
| pair3-v3, run 2 | 1.000 | 1.007 | 1.002 |

cross_entropy A/A, 15 validation pairs. Whoever ran second was 2–4.5% faster. With an odd number of pairs, one
order has one more pair, so the median lands on the majority order's side: an identical candidate read 4% faster
than itself, above a 2% promotion margin. Starting members in a different order and pinning CPU cores did not change
it; a 20 ms preheat before every window did. The cause is not identified beyond "GPU state left by the previous
window".

### 2. The wrong member was blamed for slowing the GPU

`hacks/pair/slow_opponent.py` starts a thread at import that keeps the GPU busy, so the opponent is timed on a
loaded GPU. pair3-v1 blamed the **anchor** (an infrastructure failure, no penalty for the hack): idle speed was
measured after the members imported, work queued at import was charged to whoever was checked next, and nothing was
checked after the output copy. v2 calibrates idle speed before any member exists, starts members one at a time and
checks after every wake.

### 3. A kernel could slow its own ping and read faster (pair3 and bench3)

The bench subtracts an empty round trip ("ping") from each window. A member owns its process's stdin, so it can
answer pings late while the timed work stays the same (`hacks/pair/slow_ping.py`, `hacks/attention/lateping.py`).
When each side's own ping was subtracted, every ms of delay came off its time:

| Hack | Before | After |
|---|---|---|
| pair3, rmsnorm, ping 1 ms late | pair3-v3: 0.267 (a fake 3.7× win) | pair3-v4: 0.986 |
| bench3, rmsnorm, ping 1 ms late | bench3-v2: 2.90× | bench3-v3: 0.968× |
| bench3, rmsnorm, ping 3 ms late | bench3-v2: 1514× (time clamped to ~0) | bench3-v3: 0.974× |
| bench3, attention, ping 3 ms late | bench3-v2: 1.36× | bench3-v3: 0.999× |

The fix: one floor, the fastest ping of any side, subtracted from everyone. A member can raise its own ping but not
lower the floor. bench3-v2 also clamped a non-positive time to 1e-6 s; a trainer that clips the reward would have
paid such a kernel the maximum. Both benches now fail it as infrastructure. The 16 attention fakes in `hacks/` still
score no gain on bench3-v3 with the source scan off.

pair3-v3 also showed the other side of a per-member ping: one ping the OS delayed made a window negative on an
honest copy of the incumbent (−12.7 ms). The floor fixes that too.

### 4. The member warmed up last stayed faster for the whole job

rmsnorm A/A under pair3-v4, whose 3 warm-up rounds always ran anchor, incumbent, candidate:

| Warm-up order | Members started | Validation (2 runs) | Anchor / incumbent |
|---|---|---|---|
| anchor, incumbent, **candidate** last | incumbent first | 0.991, 0.982 | 0.987, 0.992 |
| anchor, incumbent, **candidate** last | candidate first | 0.985, 0.983 | 0.985, 0.994 |
| anchor, candidate, **incumbent** last | incumbent first | 1.023, 1.014 | 1.016, 1.000 |
| incumbent, candidate, **anchor** last | incumbent first | 1.004, 1.013 | 0.985, 0.988 |
| pair3-v5: competitors alternate, **anchor** last, 4 rounds | incumbent first | 1.010, 1.007, 1.006, 1.005 (4 runs) | 0.987, 0.985, 0.983, 0.991 |

Whoever was warmed up last read 1–2.3% faster through 15 later pairs, in both pair orders; the order the processes
started in did not matter. A copy of the incumbent read 0.982, 0.2% short of a 2% promotion margin. pair3-v5 gives
the last warm-up place to the anchor, which is reported only. The mechanism is not known.

Under v5 an incumbent lead of 0.5–1.0% remains. It fits the same pattern (in the last warm-up round the incumbent
runs after the candidate) and points the safe way: a copy of the incumbent reads slower and is never promoted. A
real candidate has to beat the margin plus up to 1%.

### 5. Clock filters can drop good rows

Reading the SM clock with `nvidia-smi` after the round caught the GPU idling down: 1200–1845 MHz on equal windows.
pair3-v4 reads it through NVML under load. Even so, on newton_schulz the clock under load ranged 1365–1845 MHz, and a
filter that keeps pairs with clock ≥ 0.9 × median dropped 5 of 15 validation pairs whose times were not slow
(incumbent mean 3.63 ms dropped vs 3.66 ms kept). On this card the clock does not predict the time; the anchor's
drift is the better check.

### 6. A/A and fakes, pair3-v4

| Task (A/A) | Validation | Interaction |
|---|---|---|
| softmax | 1.000 | 1.002 |
| cross_entropy, 2 runs | 1.002, 0.998 | 1.001, 1.001 |
| attention | 0.993 | 0.998 |
| newton_schulz | 0.994 | 1.001 |
| rmsnorm | 0.986 | 0.969 (finding 4) |

| Fake (candidate, rmsnorm) | What it does | pair3-v4 |
|---|---|---|
| slow_opponent | GPU spam thread from import | invalid, blame candidate (GPU busy after its turn) |
| late_flood | waits 8 s, then floods the GPU | invalid, blame candidate (GPU busy after its turn) |
| copy_incumbent | the incumbent's own code | ok, 0.987 (finding 4) |
| wrong | zeros | invalid, blame candidate (relative error 1.0) |
| slow_ping | answers pings 1 ms late | ok, 0.986 |

`stale_cache` (returns a remembered answer) was not rerun on v4; inputs are new each round, so it can only be wrong.

## Limits

- One GPU model, one evening. swiglu ran out of memory on the 6 GB card: 3 member processes + the bench each hold
  about 1.1 GB at its size, plus a CUDA context each.
- Findings 1 and 4 are fixed by measurement, not understood. A different GPU may need a different preheat or
  warm-up; set a promotion margin from your own A/A runs (`tools/aa_report.py --gate`).
- The receipt follows the `kernel-koth-paired-v1` schema of the league that consumes it; that league and its
  clock filter are not in this folder.
