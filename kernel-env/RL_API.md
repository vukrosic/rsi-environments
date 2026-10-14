# External trainer integration contract, version 1

This is a synchronous, single-episode Python API, not a Gymnasium subclass.
The shared version-pinned RL framework adapter is maintained in `rsi/suite/`.
It must implement the mapping below and run a compatibility rollout against
the installed environment before a training claim. This package performs no
gradient updates and makes no claim that prompt comparisons train a policy.

1. Instantiate `KernelEnvironment(task, workspace, split=..., device=...,
   profile=..., seed=..., calibration=..., max_steps=..., max_episode_seconds=...)`.
   Choose a unique workspace for concurrent policies. One environment instance
   handles one episode at a time; separate instances/processes coordinate GPU
   use through the shared lease.
2. Call `reset()` and serialize the returned observation into the policy's
   context. Observation contains task/split, editable code, instruction,
   tolerance/device/profile, remaining action/time budget, and prior tool
   feedback. No final input tensors or answers appear in observations.
3. Decode policy text into one JSON action: `edit`, `test`, or `submit`, with an
   optional complete single-file `code` string. `edit` requires code. Treat
   invalid action schemas as parser failures under the trainer's policy; they
   do not bypass the environment budget by issuing unbounded retries.
4. Call `step(action)` and feed `transition["observation"]` back to the policy.
   Nonterminal rewards are 0; numeric development scores stay inside feedback.
   Stop on `terminated` or `truncated`.
5. For a terminated trajectory with `trainable=true` and
   `exclude_from_rl=false`, associate the measured terminal reward with
   the code-producing policy actions and perform the trainer's chosen RL
   update (e.g. external GRPO/PPO). For a truncated trajectory, reward is null;
   exclude it from reward learning and preserve the error/blocked record. Do
   not coerce infrastructure failure to zero, which would train on a broken
   device. Record policy checkpoint, sampling seed/config, all actions,
   episode budgets and the final candidate/scorer hashes with training logs.

The transition and its `info` score record both expose `task_split`,
`trainable`, `exclude_from_rl`, and `rl_exclusion_reasons`. Saved suite records
also retain these markers and `provenance.partition`, including when using
the raw `evaluate()` API. Eligibility requires a **terminal standard-profile
training task** with `ok` or `invalid` status. Invalid training submissions
retain numeric reward 0 and remain eligible for the correctness penalty.
Nonterminal feedback, all smoke results, all eval-only operations, and
infrastructure/execution failures are explicitly excluded. Eval-only scores
remain numeric for reporting; their numeric value never overrides the mask.
Adapters must honor the mask before building replay buffers, assigning policy
credit, or updating weights. This is a machine-readable API guard against
accidental ingestion, not protection against a trainer deliberately ignoring
the contract. Each native step persists its full transition and markers in
the episode directory as `transition-NN.json`.

Training catalog: `rmsnorm`, `softmax`, `swiglu`, `cross_entropy`.
Evaluation-only catalog: `attention`, `newton_schulz`. The constructor enforces
this split. Evaluation results must not feed training updates or lesson
selection. Public code/tasks are held out operationally, not cryptographically.
Choose disjoint trainer-seed ranges (suggested train 0–999999; eval
900000000–900999999) and log them. Within every task, final case seeds are
offset by 1 billion from development seeds and use different shapes.

Reset and later observations include `task_spec`: the signature, plain
operation definition, exact constants, nominal input/output shapes and dtypes,
baseline identity, tolerance and error formula. Shapes vary across cases;
derive dimensions from inputs rather than hardcoding the nominal dimensions.
This spec is task information, not evaluation outputs or hidden case answers.

Action count includes edits and tests. A submit or the last allowed action
evaluates current code on final cases; budget exhaustion before scoring
truncates. Each code execution is additionally bounded by `timeout`, capped
to the remaining episode wall budget. A new reset creates a unique directory
and preserves previous episode files. `final()` performs submit; it cannot
re-score a completed episode without reset.

Use smoke CPU rollouts only for installation and transition compatibility.
Research training on speed reward uses standard profiles with successful idle
calibration receipts. On this busy Mac, standard GPU timing may be blocked.
No model weights have been trained by the kernel-environment build.
