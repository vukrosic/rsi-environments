# Roadmap

Build useful reinforcement learning environments that researchers can connect to their own coding LLMs and trainers. Each environment should evaluate real optimization work, with independently measured outcomes and fixed constraints. We do not tune handcrafted puzzles to a particular agent's ability.

Updated October 15, 2026. Checkboxes describe completed deliverables, not scheduled activity. Release dates are not promised before the readiness gates pass. The [feature matrix](FEATURES.md) separates published code from development results.

## Released foundation

- [x] Publish inference, training-improvement and kernel environments as a source alpha.
- [x] Provide code-action interfaces, bounded execution, independent grading and terminal records.
- [x] Publish the Verifiers 0.4.0 adapter and compatibility examples.
- [x] Separate smoke and evaluation-only results from rewards eligible for policy training.
- [x] Document a bounded Apple GPU training check and the published release's limitations.

These milestones establish environment and compatibility behavior. They do not establish coding-policy learning, recursive improvement or external adoption.

## Next release: reproducible GPU environments

This is the immediate priority. Finish these components before expanding the environment catalog.

- [ ] Ship portable NVIDIA source-checkout setup with explicit dependencies and separately prepared assets.
- [ ] Include every required runtime helper in the public source layout; remove dependencies on private development directories.
- [ ] Verify the inference packet builder and its non-executing preflight, then check the native coding-agent terminal reward route.
- [ ] Verify the portable training-recipe runner, matched budgets, saved-weight audits and whole-job timeout/cleanup evidence.
- [ ] Ship compression source with an independent original-model reference that NVIDIA users can generate without Apple hardware.
- [ ] Verify compression's new reference contract and CUDA registration without changing historical workloads or quality anchors.
- [ ] Exercise actual framework loading and the GPU terminal reward path, including exclusion of blocked or invalid measurements.
- [ ] Publish matching source, concise reproduction commands and sanitized evidence in one new immutable release.

Kernel timing has a separate gate: the current NVIDIA host failed the existing stability threshold. Do not relax that threshold or repeat unchanged calibration to get a passing result. Standard CUDA speed rewards require qualified measurement conditions and fresh evidence; correctness diagnostics can be documented with their narrower scope.

The GPU release may explicitly exclude an unqualified lane rather than imply that every environment is ready on every GPU.

## Next: easier research integration

After the GPU release gates are met:

- [ ] Provide one minimal documented rollout-to-reward example for an external trainer, with explicit trainable/excluded episode handling.
- [ ] Expose task model and data configuration where independent grading and resource budgets can be preserved. Own-policy support and own-workload support are separate capabilities.
- [ ] Make workload, precision, memory metric, quality threshold and resource budget easy to inspect before starting a job.
- [ ] Offer compact machine-readable result summaries suitable for experiment tracking.
- [ ] Verify setup from a clean checkout with documented dependencies and no private artifacts.

The environment provides tasks and rewards. Researchers supply their optimizer; a built-in policy trainer is not required for environment completion. Any later coding-policy learning study would need its own experimental plan and evidence.

## Optimization that remains useful as agents improve

In development, versioned incumbents let a candidate challenge the best verified implementation. A confirmed winner becomes the next target. Correctness, original-model quality constraints and comparison conditions stay fixed as the incumbent changes.

- [ ] Release and document that competition interface with reproducible round creation and candidate submission.
- [ ] Retain source-bound candidate records and immutable historical results.
- [ ] Verify that incorrect code, stale submissions, source drift and noisy measurements cannot install a winner.
- [ ] Keep compatibility checks, actual promotion and learning claims as separate evidence states.

Progress may plateau. An improving score is not proof of policy learning or recursive self-improvement.

## Later environment ideas

Proposals only; these are not additional active build commitments:

| Idea | Real objective | Constraints that must remain fixed |
|---|---|---|
| Data efficiency | Reach better model quality using less training data | Evaluation data, compute budget and leakage controls |
| Request scheduling and batching | Serve more requests within a latency limit | Correct outputs, request workload, device and memory budget |
| Speculative decoding | Reduce end-to-end generation time | Required target-model output semantics and the total draft/verification cost |
| Retrieval efficiency | Search a real corpus faster or with less memory | Retrieval accuracy, corpus and query evaluation |

Select a new environment only when it adds a distinct measurable research problem and has a credible execution and grading plan.

## Contributing and release decisions

Use [GitHub issues](https://github.com/vukrosic/rsi-environments/issues) for reproducible defects, integration needs or proposed workloads. Include the relevant environment, device, dependency versions and expected versus observed result. See [CONTRIBUTING.md](CONTRIBUTING.md).

A useful release includes tested behavior, accurate documentation and an explicit account of unresolved limitations. Once a component meets its finite scope, it enters maintenance; we do not keep adding features merely to keep it active.

[Features and status](FEATURES.md) · [README](README.md) · [Releases](https://github.com/vukrosic/rsi-environments/releases)
