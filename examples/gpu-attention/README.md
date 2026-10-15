# Causal attention on NVIDIA

This example contains an exact submitted Triton kernel and its recorded result.
It computes causal forward attention for contiguous float32 CUDA tensors q, k
and v, each shaped `(8, 8, 1024, 64)` (batch, heads, tokens, head dimension).
The output has the same shape and dtype. There is no dropout or gradient calculation.

The baseline calls
`torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`.
Its selected CUDA backend and precision configuration were not recorded.

## Recorded result

| Measurement of the same source | Baseline | Candidate | Reported speedup |
|---|---:|---:|---:|
| Selected search result | 3.147 ms | 0.794 ms | 3.963× |
| Separate retime | 3.117 ms | 0.804 ms | 3.879× |

The retime passed the benchmark's correctness check: normalized maximum error
`0.0005202682223170996`, below tolerance `0.002`. This divides the maximum
absolute difference by `max(max(abs(reference)), 1e-6)`; it does not bound every
element's relative error. Both records report five timed trials, `timer: v3`,
`baseline: library` and `isolation: uid`. Rounded times need not reproduce the
reported ratios exactly.

**Precision matters:** the kernel casts its dot-product operands to float16,
while accumulating and returning float32 values. Passing this tolerance does
not establish exact float32 equivalence or accuracy on other distributions.

The code was selected from ten attempts (nine passed), then retimed once.
These are selected observations, not an average across independent runs or
evidence of LLM learning. See [receipt.json](receipt.json) for the projected records.

## Run it

Use the [existing NVIDIA timer](../../kernel-timer/README.md) on Linux with
CUDA-enabled PyTorch, Triton, NumPy and Python 3.11+. The interpreter and packages
must be accessible to its isolated worker UID. From the repository root:

```sh
cd kernel-timer
sudo python3 bench3.py attention ../examples/gpu-attention/kernel.py --baseline library --trials 5
```

This command uses fresh random inputs and the existing timer's isolated execution
setup. It matches the public CLI; a fresh dependency installation has not been
tested. Allow sufficient GPU memory for the timer's multiple processes.

## Source and measurement limits

[kernel.py](kernel.py) is byte-identical to the selected submission and retime
source. SHA256: `dfe2e4293d1fac50c4097d6dd7003af2893eeec57ce95994c5a91cbba04fe47f`.
The receipt is a privacy projection with original artifact digests, not an
authenticated execution record. Returned records omit the server source hash,
runtime hash, exact dependency versions and hardware identity.

Public timer revision
[`71c965c`](https://github.com/vukrosic/rsi-environments/tree/71c965cbb8cd618b17b92cbb1a548dbd49f32a11/kernel-timer)
is the reproduction source reference, rather than a verified pin of the original
execution. The benchmark reports corrected timing windows, rather than standalone
CUDA-event timings of one launch. Stronger baselines, other hardware, shapes,
layouts, masks and precision requirements remain unmeasured here.
