# Triton cross-entropy on NVIDIA

This example contains an exact submitted kernel and a recorded benchmark result.
It computes mean cross-entropy for CUDA logits of shape `(8192, 4096)` in float32
and 8192 int64 labels. There is no gradient calculation.

The first Triton launch computes stable log-sum-exp minus the target logit for
each row. A second launch averages those row losses. The baseline is
`torch.nn.functional.cross_entropy(logits, labels)` with mean reduction.

## Recorded result

| Measurement of the same source | Baseline | Candidate | Reported speedup |
|---|---:|---:|---:|
| Selected search result | 1.047 ms | 0.445 ms | 2.352× |
| Separate retime | 0.932 ms | 0.418 ms | 2.231× |

The retime passed the benchmark's correctness check with reported relative error
`7.5014305878085e-08`, below the task tolerance of `2e-3`. Both records report
five timed trials, `timer: v3`, `baseline: library` and `isolation: uid`.
The speedups are copied from the records; rounded times need not reproduce
the ratios exactly. See [receipt.json](receipt.json) for the projected records.

The kernel was selected from a resumed ten-attempt search and remeasured once.
These are selected observations, not an average across independent runs or
evidence that RL improved an LLM. Experiment context was an RTX A2000 6 GB.

## Run it

Use the [existing NVIDIA timer](../../kernel-timer/README.md) and its Linux,
CUDA-enabled PyTorch, Triton, NumPy and Python 3.11+ requirements. The interpreter
used below must have those packages and be accessible to the timer's worker UID.
From the repository root:

```sh
cd kernel-timer
sudo python3 bench3.py cross_entropy ../examples/gpu-cross-entropy/kernel.py --baseline library --trials 5
```

This invokes the existing benchmark with fresh random inputs. The documented
command matches its CLI; this checkout's fresh installation has not been tested.
Use the timer's documented isolated execution setup for submitted code.

## Source and measurement limits

[kernel.py](kernel.py) is byte-identical to the selected submission and retime
source. SHA256: `55aaf38ed0adb3503ba58a8a6350834f5336721d9a30b265765a08636f3e5335`.
The receipt is a privacy projection with original artifact digests, not a raw
server receipt or authenticated execution proof.

The returned records omit the runtime source hash, exact dependency versions
and hardware identity. Their `v3` label does not bind them to a particular Git
revision. Public timer revision
[`64bb641`](https://github.com/vukrosic/rsi-environments/tree/64bb64134faa0da501e1f7ca57b5b3054394ebf0/kernel-timer)
is the reproduction source reference; its hashes are in the projected receipt.

The timer reports corrected timing windows, rather than standalone CUDA-event
timings of one kernel launch. Other hardware, shapes, dtypes, strides, targets,
loss options and per-example accuracy remain unmeasured. No general speed
guarantee or promotion is claimed.
