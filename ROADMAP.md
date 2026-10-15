# Roadmap

Updated October 15, 2026. See [Features](FEATURES.md) for what is published and what has run in development.

## Published: Training Improvement alpha4

- [x] Publish the complete standalone training source and reproduction commands.
- [x] Run the documented NVIDIA training example for 500 training calls and audit its saved weights.
- [x] Run a separate native Python API episode and verify its terminal reward.
- [x] Publish matching source hashes and reviewed execution results.

See [the release](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.4)
and [verification details](examples/TRAINING-CUDA-VERIFICATION.md). These checks
establish task execution and scoring. They do not establish LLM learning or
fresh dependency installation.

## Next release: setup and a user-supplied LLM

- [ ] Complete installation into a fresh isolated NVIDIA runtime, without private dependency overlays.
- [ ] Verify one end-to-end callback example that accepts generated source and returns the native result to the caller's trainer.
- [ ] Publish the tested setup recipe, callback source and their execution limits together.

The first fresh-install attempt stopped before runtime installation because the
wheel verifier rejected vendored metadata inside a dependency. A cache-reusing
repair is in development. The callback's focused software checks have passed;
its actual native episode remains unverified. Neither result changes alpha4's
recorded NVIDIA execution evidence.

## Other environment releases

- [x] Fix inference API error classification on main and verify the matching public source with seven mocked-evaluator checks. Earlier release tags retain the defect; native timing qualification remains separate.
- [ ] Publish the reviewed kernel timing/input repairs and qualify speed rewards on NVIDIA under unchanged stability requirements.
- [ ] Publish compression with a reference-generation path that does not require Apple hardware.
- [ ] Verify compression's reference loader and CUDA registration.
- [ ] Verify request batching and retrieval on their supported workloads before publishing them as usable environments.

Kernel timing failed the stability threshold on the current NVIDIA host. Further speed-reward verification needs different measurement conditions; the threshold stays unchanged. A release will state which environments and devices passed.

## Trainer integration and configuration

- [ ] Document one rollout-to-reward example with a user-supplied LLM and trainer.
- [ ] Check GPU terminal rewards through the optional Verifiers adapter, including failure exclusions.
- [ ] Allow task models and datasets to be configured while preserving independent evaluation and budgets.
- [ ] Show workload, precision, memory metric, quality limit and budget before a run.
- [ ] Provide machine-readable result summaries for experiment tracking.

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
