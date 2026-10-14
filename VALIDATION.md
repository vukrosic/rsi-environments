# Release validation

Checked on October 14, 2026, on an Apple Silicon Mac with Python 3.12.8,
PyTorch 2.8.0, and NumPy 2.2.5. These checks used the exported public source,
not imports from the original environment folders.

| Check | Result |
|---|---|
| Shared record and resource helpers | 14 tests passed |
| Inference environment | 51 tests and 10 subtests passed |
| Training environment | 84 tests and 36 subtests passed |
| Kernel environment, before final softmax guard | 123 tests passed |
| Final softmax invariant and calibration-gate checks | 4 tests passed |
| Verifiers adapter | 26 tests passed with Verifiers 0.4.0 |
| Kernel CLI quickstart | Real CPU softmax smoke evaluation passed |
| Verifiers kernel episode | Two intercepted policy calls, native feedback, and terminal result |

The softmax guard additionally checks nonnegative probabilities and row sums.
A post-guard run passed 55 checks and timed out on the large standard CPU
calibration-gate fixture. That policy test now uses reduced real numerical
inputs with a simulated idle-host observation; the corrected gate and three
adversarial softmax checks passed. This does not certify standard timing.

Its regression cases cover outputs with missing mass, zero mass, and negative
probabilities that could pass the previous elementwise error tolerance.
The final guard checks are recorded in `release-checks.json`.

Run each native suite separately: their flat development modules can collide
when all folders are collected in one Python process.

```bash
python -m pytest suite/tests -q
(cd inference && python -m pytest tests -q)
(cd training-env && python -m pytest tests -q)
(cd kernel-env && python -m pytest tests -q)
python -m pytest suite/integration/tests -q
```

The scripted framework provider returns starter code. The kernel compatibility
episode completed with a measured native result and was correctly marked
untrainable because it used the smoke profile. A framework trace can therefore
report `ok=False` while recording successful native execution and a deliberate
reward exclusion. It is not evidence of a trained policy.

Earlier development snapshots exercised CPU framework episodes for all three
environments and their held-out tasksets. Those historical runs used separately
recorded source versions. This release's fresh framework rollout covers kernel;
it does not imply all historical episodes were rerun against this export.

One inference process-cleanup test failed during concurrent, heavily loaded
execution; the isolated supervisor suite and a complete inference rerun passed.
Process supervision samples ancestry. A detached child whose parent disappears
between observations can escape ownership discovery. These controls are not a
security sandbox and do not guarantee containment of hostile code.

No calibrated GPU speedup, outer coding-agent weight update, measured learning
gain, recursive improvement, or external adoption has been demonstrated.
Downloaded datasets and model weights are absent from the source release.
The checks reused installed dependencies; a full clean internet installation
on a second machine remains unverified.
