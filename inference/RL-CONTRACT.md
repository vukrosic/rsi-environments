# Coding-agent RL episode API v1

This environment is the execution/reward side of training a coding-agent policy.
The outer loop is model-generated Python source → environment runs source against
the pretrained model → measured terminal reward → **external trainer updates the
coding-agent model weights**. The supplied SmolLM2 is fixed task machinery, not the
coding agent being trained. Prompt comparisons or lessons are not policy training.

The suite coordinator owns the version-pinned framework adapter/training example.
This folder supplies its environment API and real-execution tests, and does not
duplicate a trainer framework. Import `CodingAgentEnvironment` from the public
`inference_env` package in both the source checkout and installed wheel.

```python
from inference_env import CodingAgentEnvironment
env = CodingAgentEnvironment('episodes/agent-policy', device='cpu', max_steps=3)
observation = env.reset(split='train', seed=14)
# An external policy/model supplies the complete Python source below.
transition = env.step({'type': 'submit', 'code': observation['starter_code']})
while not transition['terminated']:
    transition = env.step({'type':'submit',
                           'code': transition['observation']['starter_code']})
# The trainer uses the terminal reward, unless info.trainable is False.
```

`reset(split, seed)` returns JSON task state: objective, initial editable source,
action schema, task ID, correctness/resource constraints, remaining steps and
feedback. It creates a distinct trainer-owned episode directory. `seed` labels the
policy episode; fresh scorer inputs use a recorded independent nonce. Thus the
same reset seed does not replay final prompts. Exact reproduction uses the stored
inputs/source artifacts rather than requesting another fresh score.
`contract_documents` embeds UTF-8 content and SHA256 for TASK.md, DESIGN.md and
RL-CONTRACT.md. The installed wheel also ships these exact runtime resources, so
the observation's measurement and isolation rules can be resolved without a checkout.

`step(action)` accepts `{type:'submit', code:<complete Engine source>, finish?:bool}`.
It saves the source and runs the independent scorer. Intermediate actions expose
tool feedback and receive sparse reward 0. The last action, explicit `finish`, or
step-limit exhaustion runs final evaluation of that source. Terminal reward is
the suite-normalized measured reward. Syntax/invalid actions consume the step
budget; their terminal reward is zero. An infrastructure or timing failure ends
the episode with null reward and `info.trainable=False`, so the trainer must
exclude it from policy updates. Calling step after termination requires reset.
The response is a dictionary with `observation`, `reward`, `terminated`,
`truncated`, and `info`; no Gym/Verifiers dependency is imposed here.

## Training and held-out evaluation split

`tasks.json` freezes separate task IDs and non-overlapping text theme sets.
`split='train'` uses training themes in development **and terminal scoring**;
terminal prompts are fresh instances, never a held-out evaluation reward used
for training. It allows 1–10 source actions, configured before reset.
`split='heldout'` uses its separate themes, one terminal action, and always sets
`info.trainable=False`. Held-out rewards are reporting metrics only. Both splits
retain random-token stress cases and the same model/prompt lengths. This is a
small vocabulary-family holdout, not evidence of transfer to new models or systems.

The earlier research-loop `InferenceEnvironment` remains available. Its default
final mode uses held-out themes, suitable for development/final research separation.
It must not be used as an outer RL training reward source without an explicit
training partition. The new CodingAgentEnvironment routes partitions correctly.

## Adapter requirements and verification boundary

The adapter must keep episode directories and score tools outside editable agent
workspaces, translate model tool calls to this action schema, supply source strings,
pass observations/feedback back to the policy, preserve horizon/done state, and
train only on terminal train-partition transitions with finite valid rewards.
Invalid candidate terminal reward 0 is a training example; blocked/error null is
an excluded infrastructure sample. Do not replace null with 0. Evaluation episodes
must be isolated from training updates. CPU/MPS measurements cannot be mixed.

Use the shared accelerator lease already inside scoring, and give trainer/tool
timeouts above the 600-second controller budget to allow result delivery and cleanup.
The parent supervisor bounds preflight, loading, snapshots and grading as well as
workers; host filesystem failures in the outer API still require an external supervisor.
Trusted local source only;
hostile code needs the external sandbox described in `DESIGN.md`. An adapter
compatibility test is distinct from actual agent-weight training and from a
demonstrated learning gain. See [root release validation](../VALIDATION.md) for exactly which were verified here.
