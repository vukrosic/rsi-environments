# Shared coding-agent integration (Verifiers 0.4.0)

This package adapts the inference, TinyStories-training and kernel environment
APIs to **Prime Intellect Verifiers v1**. It exports a typed `CodingTaskset`,
`CodingHarness`, and `CodingEnv`; `vf.load_environment` resolves all three through
its real plugin loader. This is the released 0.4.0 API, not the removed legacy
`MultiTurnEnv` interface.

Verified locally: actual framework loading, interception/model-call traces,
generated-source actions, native feedback, and terminal receipts for all three
CPU environments. The compatibility provider copies starter code. **No outer
coding-policy weights were trained and no learned improvement was demonstrated.**
Standard accelerator reward calibration remains a separate checkpoint.
See [release validation](../../VALIDATION.md) for tag-specific source checks and their limitations.

## Install without changing global packages

From the repository root, using Python 3.12:

```sh
python3.12 -m venv suite/integration/.venv
suite/integration/.venv/bin/python -m pip install -r suite/integration/requirements-framework.lock
suite/integration/.venv/bin/python -m pip install -r suite/integration/requirements-runtime.txt
suite/integration/.venv/bin/python -m pip install --no-deps -e suite/integration
suite/integration/.venv/bin/python -m pytest suite/integration/tests -q
suite/integration/.venv/bin/python suite/integration/smoke.py --environments kernel --output runs/kernel-compatibility
```

`requirements-framework.lock` pins packages installed inside the isolated venv.
`requirements-runtime.txt` pins the scientific/runtime dependencies reused from
the existing environment. A fresh machine must install both specifications;
this Mac reused torch 2.8.0 and downloaded no replacement torch. Large assets and private development traces are excluded from this source release.
Install each native environment's remaining pinned dependencies/assets as
specified by its own README. Inference uses the already-cached, hash-checked
SmolLM2-135M; training uses `training-env/data` with a real whole-story manifest.
The integration does not silently download task assets.

The editable package expects the repository source tree by default. For a relocated
checkout, set `env.agent.harness.checkout`, `data_dir`, and `output_dir` explicitly.
For separate worker execution, set the harness `python` to the matching runtime.

## Mapping and boundaries

One JSON model response contains exactly `type` and `code`. `code` is a complete
Python source string. `test` and `submit` apply to all three tasks; kernel also
supports `edit`. The harness translates these into each native schema. Native
observations/starter source and subsequent tool feedback become policy messages.
Filesystem paths and scorer provenance are kept in operator evidence, not the
policy context. All model calls go through Verifiers' interception endpoint and
therefore become real framework `ModelCall`/message graph records.

Each native episode runs in a separate process to avoid colliding flat module
names (`worker`, `data`, `evaluate`). A UUID directory holds policy responses,
source actions, transitions, frozen interface/scorer sources, manifests and
native evaluation evidence. Native evaluators retain their own immutable code
snapshots, metrics and normalized suite envelopes. Source drift during a rollout
adds a training exclusion. This subprocess boundary is for trusted local code;
it is not a hostile-code sandbox. Use external isolated execution before
accepting adversarial submissions.

Every model response consumes one of `horizon` turns, including malformed JSON,
unknown keys/types, duplicates, missing/blank/overlong source and schema failures.
The last accepted action is forced to terminal submission. A malformed final
response yields an explicit parser-invalid zero receipt, marked as **unmeasured**.
It does not fabricate a scorer result. Model output tokens, request time, native
tool time and episode wall time are bounded separately. Native syntax/incorrect
candidate failures remain real zero-reward cases; device/execution/blocked/null
failures remain excluded. After terminal/truncated state, the loop ends and the
native process is closed. A new rollout always allocates a new native episode.

An RL reward is exposed only for a finite, valid, terminal **train** receipt in
the **standard** profile, without a native exclusion or invalid timing marker.
Heldout/evaluation-only tasks, final TinyStories data, smoke/diagnostic scores,
truncated episodes and blocked/error/null receipts cannot feed policy updates.
For excluded episodes the reward hook raises a deliberate scoring exclusion;
Verifiers retains the native receipt in `trace.info['rsi']`, leaves the reward
unset, marks the trace failed, and our finalizer sets `trace.agent.trainable=False`.
Thus compatibility runs can correctly have `episode.ok=False`: that is the
training exclusion, not evidence that the native action/feedback failed.
An external trainer must filter failed/untrainable episodes, not read
`trace.reward`'s empty-sum zero as a training label.

Train/heldout task IDs and seed ranges are disjoint. Kernel train tasks are
rmsnorm/softmax/swiglu/cross_entropy; the current taskset samples rmsnorm.
Heldout kernel uses attention. Training heldout uses seeds 10000–19999; kernel
heldout uses 900000000+; inference uses separate theme/task IDs.

All task-side accelerator work continues through the native suite lease,
`~/.cache/vuk-research-envs/accelerator.lock`. GPU trainers must acquire the same cooperative lease for sampling and
optimization, releasing it before native GPU tools need that lease. Do not hold
an outer lease across a native tool call (nested acquisitions can deadlock).
Other apps do not honor this lock; idle calibration still matters.

## Runnable outside-trainer entry points

Use `eval.toml` with an existing **local** OpenAI-compatible coding-policy server:

```sh
suite/integration/.venv/bin/vf-eval @ suite/integration/eval.toml --dry-run
suite/integration/.venv/bin/vf-eval @ suite/integration/eval.toml
```

The checked config keeps publication disabled and concurrency one. It is a
compatibility config. For reward training, use `profile='standard'`,
`compatibility_only=False`, the native matching calibration receipt where
required, sufficient policy context/output budget, and a stable source snapshot.
`vf.TrainClientConfig` is the 0.4.0 training-client path to a local vLLM
`/inference/v1/generate` server: Verifiers renders policy tokens/logprobs for an
external optimizer. That vLLM/prime-rl path was not executed on this Mac.

The first alpha supplies the environment adapter and a scripted compatibility
runner. It does not ship the experimental policy optimizer. Use a separately
validated trainer for coding-agent weight updates, and evaluate any learning
claim against an untouched held-out taskset. Inner TinyStories GPT updates are
task machinery, not outer coding-policy training.

## Primary API references

- [Verifiers 0.4.0 source](https://github.com/PrimeIntellect-ai/verifiers/tree/6e69354ffe1b445603b493b722d0ecffbca6686d)
- [Pinned taskset API](https://github.com/PrimeIntellect-ai/verifiers/blob/6e69354ffe1b445603b493b722d0ecffbca6686d/docs/v1/tasksets.md)
- [Pinned in-process harness/interception contract](https://github.com/PrimeIntellect-ai/verifiers/blob/6e69354ffe1b445603b493b722d0ecffbca6686d/verifiers/v1/harness.py)
- [Prime-rl environment-source configuration](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/configuration.md)

Only Verifiers 0.4.0 is the tested/pinned framework here. Prime-rl's current source
configuration is an external deployment reference, not a claimed optimizer run
or compatibility pin for this local checkpoint.
