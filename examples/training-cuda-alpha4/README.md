# Training Improvement

An LLM submits Python training code. The environment runs the task model,
independently scores its prediction loss, and returns an eligible reward to your
RL trainer. Use it to experiment with improvements to training code. You supply
the LLM and trainer.

## Quick start

Extract the source archive and keep `training-env/` and `suite/` as siblings.
Use your existing compatible Python environment; follow
[INSTALL.md](training-env/INSTALL.md) for dependencies and optional overlays.
Prepare fresh TinyStories data with `training-env/prepare_public_data.py`, as
shown in [RAW-DATA.md](training-env/RAW-DATA.md).

Run the following from the extracted root. Set `CUDA_PYTHON` to your interpreter
and `DATA` to the prepared `data` directory. Set `RUN` and `SESSION` to distinct,
fresh directories outside the checkout and data cache. Create `RUN` first.
Keep the GPU idle while these commands run.

```sh
"$CUDA_PYTHON" -B training-env/reward_control.py --output "$RUN/source-control.json"
"$CUDA_PYTHON" -B training-env/calibrate.py --device cuda --idle-confirmed \
  --output "$RUN/calibration.json"
"$CUDA_PYTHON" -B training-env/interaction.py reset --session "$SESSION" --data "$DATA" \
  --profile standard --device cuda --seed 14 --task-split train --max-trials 1 \
  --calibration "$RUN/calibration.json" > "$RUN/reset.json"
"$CUDA_PYTHON" -B training-env/interaction.py final --session "$SESSION" --data "$DATA" \
  --candidate training-env/starter.py --profile standard --device cuda --seed 14 \
  --task-split train --max-trials 1 --calibration "$RUN/calibration.json" \
  > "$RUN/transition.json"
"$CUDA_PYTHON" -B training-env/reward_control.py --transition "$RUN/transition.json" \
  --data-manifest "$DATA/manifest.json" --output "$RUN/reward-control.json"
```

Run the commands sequentially and stop if one fails. This example submits
`training-env/starter.py` and writes the terminal response to `transition.json`.
The final command checks that response against the documented starter, source
files, prepared data, task counts, and reward formula. CUDA execution of this
release is pending verification.

## Submit code from your LLM

Your LLM must produce complete Python source implementing `build_model()`,
`build_optimizer(model)`, and `train_step(model, opt, x, y, frac)`. Start from
`training-env/starter.py`. From `training-env/`, use the Python API:

```python
from interaction import TrainingEnv

env = TrainingEnv(session_path, data_path, profile="standard", device="cuda",
                  seed=14, task_split="train", max_trials=1,
                  calibration=calibration_path)
observation = env.reset()
# Give observation to your LLM and obtain its complete source as a string.
transition = env.step({"code": generated_source, "final": True})
```

Use a new session for this submission. The response contains the loss, reward,
termination state, eligibility fields, and native record. Your trainer consumes
eligible terminal training rewards. For a successful submission, reward equals
`exp(-loss)`. Keep candidate-invalid zero rewards and exclude infrastructure
failures, which have null reward. Exclude nonterminal feedback, smoke tests,
and held-out evaluation from policy updates. See
[RL-INTEGRATION.md](training-env/RL-INTEGRATION.md) for the complete contract.
The starter-specific `reward_control.py` command above does not validate an
arbitrary LLM submission.

## Task and hardware

The supplied task trains a fresh 5,063,424-parameter GPT on TinyStories text,
with vocabulary 4,096, context 256, and batch size 32. Standard mode makes 500
training calls using 4,096,000 tokens, then independently grades 491,520 dev
predictions. The supplied model and data preparation define this task; arbitrary
replacement workloads are outside this release.

The observed runtime is Linux with an NVIDIA RTX A2000, Python 3.12.14,
Torch 2.11.0+cu128, NumPy 2.5.3, and tokenizers 0.22.1. The source checkout is
the installation; a training wheel is unnecessary. Fresh dependency and driver
installation have not been verified. The public CUDA example permits an attempt
on another single NVIDIA device, but execution there remains unverified.

`training-env/public_cuda_example.py` offers a separate command that prepares
public data, trains the task model, independently evaluates its saved weights,
and audits those weights. Follow [PUBLIC-CUDA.md](training-env/PUBLIC-CUDA.md).
Its evaluator and weight audit are separate from the native `TrainingEnv`
reset/final interaction above. Both still await end-to-end verification for this
release. Historical losses and verified raw data preparation do not establish
a successful CUDA run of these source files. We have not trained your outer LLM
or demonstrated a learning improvement.

## Running an experiment

Keep the same dependency overlays for every native command and its subprocesses.
All cooperating GPU jobs should use the common accelerator lock, whose default
is `~/.cache/vuk-research-envs/accelerator.lock`. You can set
`RESEARCH_ENVS_ACCELERATOR_LOCK` to another absolute path shared by those jobs.
Calibration and evaluation acquire it internally; do not hold it around them.
Keep data, job outputs, and checkpoints outside the source checkout.

Use an external supervisor to bound calibration to 90 seconds, reset to 15
seconds, final to 900 seconds, and the complete job to 1,050 seconds, with bounded
cleanup of its subprocesses. Native training has a 300-second training cap;
worker and grader each have a 600-second child timeout. These controls do not
impose a hard aggregate RAM or VRAM quota. Preserve calibration, source and data
identities, native records, and cleanup results when reviewing a reward.
Do not turn a timeout or missing infrastructure evidence into candidate zero.

Submitted Python runs with your permissions. Process controls are not a security
sandbox; use isolated execution for untrusted code. Source is MIT licensed.
Downloaded data retains its publisher's terms.
