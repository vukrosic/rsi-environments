# Inference optimization environment

An RL coding-agent environment: a model submits inference code, reads execution
feedback, and receives a terminal measured reward. An external RL trainer updates
the coding agent’s weights. The fixed task model is pretrained SmolLM2-135M. The comparison baseline uses SDPA, a KV cache, and a
last-token output projection. Every timed request is checked for correct logits
and exact greedy tokens. Incorrect outputs receive zero reward.

```bash
cd inference
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python assets.py
.venv/bin/python scorer.py solution.py --episode episodes/my-run --device cpu
```

Use `--device mps` for the Mac GPU. Use `--diagnostic` while the machine is busy:
it verifies correctness but provides no timing reward. Read `record.json` or use
the [Python interface](interface.py); blocked/error records have null rewards.

- [Coding-agent reset/action/termination API and training split](RL-CONTRACT.md)
- [Installation and reproduction](REPRODUCE.md)
- [Recheck a saved run without inference](replay.py)
- [Task, numerical checks, reward, and resource budgets](TASK.md)
- [Measurement protocol and trust boundary](DESIGN.md)
- [Release checks and limitations](../VALIDATION.md)

This is a local release candidate for trusted research code. It has a shared [Verifiers adapter](../suite/integration/README.md). A hostile-code sandbox and CUDA backend are not verified.
The other training and kernel environments are not needed to run this folder.
