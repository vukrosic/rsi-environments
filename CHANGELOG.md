# Change log

## v0.1.0-alpha.3 — October 14, 2026

- Record one independently reviewed real TinyStories inner-GPT smoke on Apple MPS: 5,063,424 parameters, 3 scheduled training calls (2 with positive learning rate), 96 training tokens and 64 development grading tokens.
- Publish sanitized protocol, device-boundary evidence, source/data/weight hashes and a public CLI reproduction command. The observation wrappers were independently checked; the public command repeats the uninstrumented workload and does not recreate the wrapper audit.
- Add a [NVIDIA training guide](docs/nvidia-training.md) and exact Mac-validated Prime-RL example configs; actual NVIDIA execution remains pending.
- Keep scope explicit: no before/after learning gain, standard CUDA execution or coding-agent RL update is established. See [VALIDATION.md](VALIDATION.md) and [the Apple GPU check](docs/apple-gpu-check.md).

## v0.1.0-alpha.2 — October 14, 2026

- Make the inference task rules/probe contract available in installed packages; tighten cache probes, infrastructure error attribution and lazy CPU thread limits.
- Add a versioned real-data CPU training diagnostic, split/content integrity checks and cooperative execution bounds. Saved inner-GPT updates and paired losses were checked separately from coding-policy training.
- Route kernel calibration by difficulty and retain portable package helpers and licenses. Unreliable standard timing remains excluded.
- Repair adapter cancellation cleanup and include its detached-Python ownership hook in the public source.
- Restore the shared source dependency used by standalone training, and separate historical alpha1 evidence from new export checks.

This remains a limited CPU alpha. The public adapter ships compatibility execution, not the experimental coding-policy optimizer. No standard accelerator reward, coding-policy update or held-out learning gain is claimed.

## v0.1.0-alpha.1 — October 14, 2026

First public alpha of the environment and reward layer for LLM self-improvement research.

### Added

- Inference optimization against a fixed pretrained SmolLM2-135M, with cached baseline, output checks and diagnostic timing mode.
- Real TinyStories training-improvement task with independent loss scoring and separate development/final data.
- Six numerical kernel tasks with competitive baselines, correctness checks and separate training/evaluation operations.
- Budgeted code-action episodes, feedback, terminal rewards and machine-readable RL eligibility.
- Shared finite score records and cooperative accelerator lease.
- Verifiers 0.4.0 typed adapter, compatibility runner and local policy-server configuration.
- Source-only distribution with datasets, weights, dependency environments and private runs excluded.

### Scope

CPU task/framework compatibility is recorded. Coding-policy weight training, learned improvement, valid standard accelerator rewards and hostile-code isolation are not release claims. The experimental local optimizer is omitted pending independent-review fixes.

Later release notes will separate implemented features, fixed defects, executed checks and remaining limitations. A development checkpoint is not a measured improvement in the coding policy.
