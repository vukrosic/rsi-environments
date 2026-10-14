# Training Improvement Environment

An RL coding agent submits model/optimizer/training-step code, receives measured
tool feedback, and ends an episode with independently scored reward. The starter is
a 5.06M-parameter GPT on TinyStories.

```bash
# From this folder, after installing requirements.txt (see INSTALL.md):
python prepare.py --synthetic --output smoke-data
python interaction.py step --session runs/demo --data smoke-data \
  --candidate starter.py --profile smoke --device cpu
python interaction.py final --session runs/demo --data smoke-data \
  --candidate starter.py --profile smoke --device cpu
```

Smoke checks plumbing only. For real data and standard runs, follow
[INSTALL.md](INSTALL.md) and [PROTOCOL.md](PROTOCOL.md). The Python policy interface is `TrainingEnv.reset()` and
`.step({"code": source, "final": boolean})`; source-path CLI calls are also supported.
[RL-INTEGRATION.md](RL-INTEGRATION.md) defines the two-model loop and trainer contract.
Each session allows at most 10 dev submissions and one final submission.
Every evaluated source is frozen; normalized suite records contain measured
loss, reward `exp(-loss)`, hashes, seeds, dependencies, hardware and conditions.

[release validation](../VALIDATION.md) records what was actually run.
This is cooperative-code evaluation, not a hardened sandbox for hostile Python.
