# Start here

These environments give an LLM a programming task, run its submission and return a measured result. You supply the LLM and the RL trainer that consumes eligible rewards.

The [Linux software checks](LINUX-CHECKS.md) verify selected packaging and result-handling behavior on GitHub runners. They are separate from the GPU execution results below.

## Choose a starting point

| Your goal | Start with | What has run |
|---|---|---|
| Train an LLM to improve training code | [Training Improvement CUDA example](../examples/training-cuda-alpha4/README.md) | Two NVIDIA runs of 500 training calls, including a separate native API episode with a terminal reward. |
| Train an LLM to write faster kernels | [NVIDIA kernel benchmark](../kernel-timer/README.md) | RTX A2000 development measurements. Timing rewards and winner promotion are currently excluded: see the [timer advisory](../kernel-timer/KNOWN-LIMITATIONS.md). |
| Connect the earlier environment APIs to Verifiers | [Shared adapter](../suite/integration/README.md) | Framework compatibility checks. The published CUDA training example uses its native API; GPU reward flow through this adapter remains to be verified. |

## Training Improvement

The LLM writes Python that defines a model, optimizer and training step. The environment trains a separate task model within a fixed budget, then another process grades its prediction loss. A successful terminal reward is `exp(-loss)`.

1. Read the [dated CUDA results and tested runtime](../examples/TRAINING-CUDA-VERIFICATION.md).
2. Follow the [complete example's setup and commands](../examples/training-cuda-alpha4/README.md).
3. Follow the [Python API guide](../examples/training-cuda-alpha4/training-env/RL-INTEGRATION.md) to submit the source your LLM generated.
4. Pass only eligible terminal training rewards to your trainer. Preserve candidate-invalid zero rewards; exclude infrastructure failures with null rewards.

The model trained inside the task is different from the LLM your trainer updates. The supplied task currently fixes its model and data. Arbitrary task-model and dataset replacements are not supported by this release.

The tested source bundle is immutable. Its bundled instructions retain their pre-run status wording; the [dated verification page](../examples/TRAINING-CUDA-VERIFICATION.md) records the subsequent successful runs. Fresh dependency installation and other GPU models remain unverified.

## Kernel optimization

The LLM writes a kernel implementation. The standalone benchmark checks its outputs and compares its runtime with the fastest measured baseline selected for that task: a library operation or reference implementation, with or without compilation. Its JSON result distinguishes candidate failures from infrastructure failures. The published v5 timer has source-discovered predictable timed inputs and an incomplete fingerprint-length check. Keep its timing rewards and winner promotion disabled until these gaps are repaired and the replacement is qualified. The review found possible paths; no historical exploitation is demonstrated. See the [timer advisory](../kernel-timer/KNOWN-LIMITATIONS.md) for details.

Use the commands and runtime requirements in the [kernel benchmark guide](../kernel-timer/README.md). Inspect its `ok`, `speedup`, `fault`, `stage` and `isolation` fields when connecting a trainer. The benchmark does not include an outer LLM optimizer. Its process isolation and verification limits are documented alongside the measurements.

The older [kernel environment API](../kernel-env/README.md) has separate timing advisories. Publishing the new benchmark does not qualify that older reward path.

## What the published results establish

Training and scoring the task model have run. The published examples do not demonstrate that an outer LLM learned from these rewards or improved on held-out tasks.

Model compression and the other environments under development are not included in alpha4. See [Features](../FEATURES.md) for current status and [Roadmap](../ROADMAP.md) for the remaining work.

For installation problems or an integration request, open a [GitHub issue](https://github.com/vukrosic/rsi-environments/issues) with the command, source revision, runtime and failure record. Remove credentials and private paths before sharing logs.
