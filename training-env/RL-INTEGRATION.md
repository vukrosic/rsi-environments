# Coding-agent RL integration: training-episode-v1

This folder implements the **environment**, not an agent-policy optimizer. There
are two distinct models:

1. The candidate small GPT learns from TinyStories during every evaluation.
   Its held-out next-token loss defines the task reward.
2. The external coding/research policy produces candidate Python. A shared RL
   trainer consumes episodes/rewards and updates that policy's weights.

An edit to GPT learning rate, a prompt comparison, or an agent reading prior
notes is not outer-policy model training. No outer policy was trained here.
The coordinating chat owns the version-pinned framework adapter and trainer;
this folder supplies and tests the environment-side contract only.

## Calls and exact contract

```python
from interaction import TrainingEnv

env = TrainingEnv('runs/agent-001', 'data', profile='standard', device='mps',
                  seed=0, task_split='train', calibration='runs/idle-mps.json')
observation = env.reset()  # task, constraints, starter source, budget, action schema
feedback = env.step({'code': policy_generated_source, 'final': False})
# policy consumes feedback['observation'], edits code, and finally:
transition = env.step({'code': policy_generated_final_source, 'final': True})
```

`reset()` returns a JSON-compatible observation dictionary. `step()` returns:

```text
observation: {status, held_out_loss, reason, measured_dev_reward,
              trials_remaining, requires_terminal_action}
reward: float in [0,1] at terminal, otherwise null
terminated: bool, true for a submitted final action
truncated: bool, true for infrastructure blocks/execution failures
info: {api_version, task_split, record_path, exclude_from_rl, scored_data_split}
record: normalized suite envelope with measured raw metrics/provenance
```

A dev action is tool feedback with no RL reward. If a framework requires numeric
rewards each step, adapt nonterminal null to 0. A valid terminal action receives
`exp(-held_out_loss)`; invalid candidate code receives 0. Blocked/error results
have null reward and `exclude_from_rl=True`: discard/retry as a **new** episode,
never map these failures to a training reward. Truncation closes the session.
Null terminal reward is therefore not a zero reward. An invalid action envelope
(type, unknown keys, blank/overlong code) raises ValueError before consuming a
trial; the adapter should reject or penalize such malformed policy outputs.

The policy action is complete source text, at most 256,000 UTF-8 bytes, not a
loss number, checkpoint, or privileged shell command. The environment writes
an action file and freezes a copy for each evaluation. Candidate stdout is not
scoring input; the tool feedback is evaluator-computed loss or a failure reason.
Each episode gets ≤10 intermediate tries plus exactly one terminal submission.
After ten dev tries, feedback requests a final action. Reset/reopening preserves
the budget. The trainer enforces a bounded token/time budget for policy generation
separately; none is claimed by this environment.

## Separate policy training and held-out evaluation

| Outer episode | Seed IDs | Intermediate feedback | Terminal reward data | Trainer use |
|---|---|---|---|---|
| `task_split='train'` | 0–9999 | dev | dev | update policy weights |
| `task_split='heldout'` | 10000–19999 | dev | final | evaluation only; never update policy |

All inner models train only on `train.npy`. `dev.npy` and `final.npy` contain
whole disjoint stories and are checked at preparation. Outer task IDs differ by
initialization/batch seed: this release is one training-improvement task family,
not a demonstrated suite of unrelated research tasks. Final data is never used
for terminal **training** rewards. Evaluation agents can iterate on dev, then
make one terminal final submission. Evaluation can expose final reward to the
operator, but the trainer must exclude it from updates and training replay.

New episodes need new session directories and immutable task configuration.
The adapter owns allocation of tasks/episode IDs, policy rollout sampling,
trajectory storage, tokenization/logprobs, batching, advantage calculation,
optimizer/distributed updates, checkpoints and validation. Bind only the
observation/action/tool result to the agent context; keep full record paths and
final assets operator-side. Our Python contract is deliberately framework-neutral;
it is not claimed to implement Gymnasium's tuple API or any framework's adapter.

## Compatibility verification scope

CPU tests exercise code-string actions through real candidate training and
independent scoring, transition fields, terminal measured reward, training/dev
versus heldout/final routing, seed partitions, budget persistence, invalid syntax,
and truncated/no-reward failure handling. [release validation](../VALIDATION.md)
records the concrete run. This verifies the environment contract on the available
runtime. A specific RL framework integration or outer weight update requires the
coordinator's adapter test and is **not verified here**.

## Additive API declarations — October 14, 2026

`info.trainable` is true only for standard terminal training episodes with
`ok`/`invalid` status. `info.reward_eligibility_reasons` explains nonterminal,
smoke, heldout or infrastructure exclusions. Measured reward is preserved even
when ineligible. This declaration complements the coordinator's stricter adapter
checks; it does not authorize policy updates on smoke or final data.
`reset().last_feedback` restores the previous measured observation on reopening
a persistent episode. The action schema and `training-episode-v1` API remain
compatible. `compatibility_check.py` checks the actual shared backend's JSON-lines
reset/test/submit route and its smoke exclusion, writing only local evidence. It
does not claim a Verifiers rollout or an outer model optimizer update.

The continuous checkpoint includes actual Verifiers 0.4.0 episodes with a local
scripted source provider. No agent policy weights are trained. The shared reward
hook deliberately reports scoring exclusion errors for smoke, heldout and null
infrastructure samples; framework `ok=false` is therefore expected for those
checks, while native termination/truncation and `agent.trainable=false` are
asserted. Compatibility proof is distinct from a successful trainable rollout.
A live-source drift exclusion is preserved; subsequent checks freeze needed
native source so parallel builder edits cannot contaminate their snapshots.
