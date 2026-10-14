# Change log

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
