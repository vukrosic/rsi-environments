# Roadmap

Updated October 16, 2026. See [Features](FEATURES.md) for what is published and what has run in development.

## Published: Training Improvement alpha4

- [x] Publish the complete standalone training source and reproduction commands.
- [x] Run the documented NVIDIA training example for 500 training calls and audit its saved weights.
- [x] Run a separate native Python API episode and verify its terminal reward.
- [x] Publish matching source hashes and reviewed execution results.

See [the release](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.4)
and [verification details](examples/TRAINING-CUDA-VERIFICATION.md). These checks
establish task execution and scoring. They do not establish LLM learning or
fresh dependency installation.

## Next release: portable setup and offline verification

- [ ] Publish the portable Training Improvement bundle with its runtime source dependencies and prepared-data requirements.
- [ ] Verify source loading on CPU, training on a supported GPU and installation in a fresh isolated NVIDIA runtime without private dependency overlays.
- [ ] Publish an offline CLI that checks saved receipts and source hashes without rerunning training.

The new bundle is prepared, but its CPU execution, supported-GPU run and fresh
isolated NVIDIA installation remain pending. Fifteen synthetic
offline CLI checks passed, but the CLI is not published and the actual-receipt offline
phase has not run. Three direct-store metadata fixture controls passed, but they
do not establish a native API episode or published direct-store API. Alpha4's
existing runs do not verify this new bundle or a fresh installation.

## Other environment releases

- [x] Publish the standalone NVIDIA kernel benchmark with library baselines and RTX A2000 control results. This is separate from the older kernel environment API and its timing advisories.
- [x] Fix inference API error classification on main and verify the matching public source with seven mocked-evaluator checks. Earlier release tags retain the defect; native timing qualification remains separate.
- [ ] Publish the reviewed kernel timing/input repairs and qualify speed rewards on NVIDIA under unchanged stability requirements.
- [ ] Publish compression with a reference-generation path that does not require Apple hardware.
- [ ] Verify compression's reference loader and CUDA registration.
- [ ] Verify request batching and retrieval on their supported workloads before publishing them as usable environments.

Kernel timing failed the stability threshold on the current NVIDIA host. Further speed-reward verification needs different measurement conditions; the threshold stays unchanged. A release will state which environments and devices passed.

## Easier research integration

- [x] Run the [public Linux software workflow](docs/LINUX-CHECKS.md) for packaging, reward-record handling and process cleanup without model downloads.
- [x] Add structured issue forms for installation failures and trainer integration requests.

- [x] Publish a [starting guide](docs/GETTING-STARTED.md) that separates runnable examples, tested runtimes and trainer responsibilities.
- [ ] Publish and CPU-check a direct `Episode` API that uses the caller's explicitly supplied, trusted configured store.
- [ ] Verify a generated-source callback submission on GPU and document the native result returned to the caller's trainer.
- [ ] Check GPU terminal rewards through the optional Verifiers adapter, including failure exclusions.
- [ ] Allow task models and datasets to be configured while preserving independent evaluation and budgets.
- [ ] Show workload, precision, memory metric, quality limit and budget before a run.
- [ ] Provide machine-readable result summaries for experiment tracking.

The direct-store binding is prepared in source. Its three metadata fixture
controls passed, but these controls do not establish the published API or a
native episode. The callback's software checks have passed, but its native GPU
episode remains unverified.

## Competing against the best result

- [ ] Publish the versioned-incumbent interface for creating rounds and submitting candidates.
- [ ] Verify that only a correct, qualifying improvement replaces the incumbent.
- [ ] Preserve the original quality limit as the incumbent changes.
- [ ] Reject stale submissions, changed source and unreliable measurements.

## Possible later environments

| Environment | Objective |
|---|---|
| Data efficiency | Improve quality with fewer training examples or tokens. |
| Speculative decoding | Reduce generation time, including draft and verification costs. |

These are proposals. Request batching and retrieval are already in development;
their release checks are listed above. Finishing usable releases takes priority
over adding another environment.

Report defects or request integrations through [GitHub issues](https://github.com/vukrosic/rsi-environments/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for reproduction details to include.
