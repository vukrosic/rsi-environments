# Roadmap

Updated October 15, 2026. See [Features](FEATURES.md) for what is published and what has run in development.

## Next release: GPU setup and reproduction

- [ ] Run NVIDIA setup from a clean checkout with documented dependencies and assets.
- [ ] Include every required runtime file in the public package.
- [ ] Verify inference packet creation and terminal rewards through its Python API.
- [ ] Verify the portable training runner, saved-weight checks and job timeout.
- [ ] Publish compression with a reference-generation path that does not require Apple hardware.
- [ ] Verify compression's reference loader and CUDA registration.
- [ ] Check GPU terminal rewards through the Verifiers adapter, including failure exclusions.
- [ ] Publish matching source, reproduction commands and execution results together.

Kernel timing failed the stability threshold on the current NVIDIA host. Further speed-reward verification needs different measurement conditions; the threshold stays unchanged. A release will state which environments and devices passed.

## Trainer integration and configuration

- [ ] Document one rollout-to-reward example with a user-supplied LLM and trainer.
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
| Request scheduling and batching | Increase throughput while meeting latency and memory limits. |
| Speculative decoding | Reduce generation time, including draft and verification costs. |
| Retrieval efficiency | Reduce search time or memory while preserving retrieval accuracy. |

These are proposals. The current environments and GPU release take priority.

Report defects or request integrations through [GitHub issues](https://github.com/vukrosic/rsi-environments/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for reproduction details to include.
