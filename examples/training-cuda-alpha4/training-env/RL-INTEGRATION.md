# Native LLM policy interface

From `training-env/` in the extracted checkout:

```python
from interaction import TrainingEnv

env = TrainingEnv(session_path, data_path, profile="standard", device="cuda",
                  seed=14, task_split="train", max_trials=1,
                  calibration=calibration_path)
observation = env.reset()
transition = env.step({"code": policy_generated_source, "final": True})
```

`reset()` returns task constraints, starter code and remaining budget.
The source-string action is at most 256,000 UTF-8 bytes. `env.final(source_path)`
is the convenience API used by the [documented native CLI](../README.md).
`step()`/`final()` returns observation, reward, terminated, truncated, info and
the full native record. Source paths and records remain operator-side.

Use only successful standard terminal training episodes on dev with
`info.trainable=true`, `exclude_from_rl=false`, `terminated=true` and
`truncated=false`. Check the native record and control evidence too. Their
reward is `exp(-held_out_loss)`. Candidate-invalid results retain native zero;
infrastructure failures retain null reward and must be excluded. Nonterminal
feedback is not terminal reward. Smoke and heldout results never update policy.
Training seeds are 0–9999; heldout seeds 10000–19999. Heldout terminal actions
score final data for evaluation only. The documented training action scores dev;
it does not turn final data into a policy reward.

You supply the outer LLM, source sampling, rollout/token budgets, optimizer,
checkpoints and independent policy evaluation. The fixed inner GPT is trained
inside each task; this is distinct from updating the LLM that writes code.
This export demonstrates no outer policy update or research improvement.
A native API receipt is separate from evaluator/weight-audit completion and from
independent session acceptance. See [PUBLIC-CUDA.md](PUBLIC-CUDA.md).
