# Inference optimization environment

An RL coding-agent environment: a model submits inference code, reads execution
feedback, and receives a terminal measured reward. An external RL trainer updates
the coding agent’s weights. The fixed task model is pretrained SmolLM2-135M. The comparison baseline uses SDPA, a KV cache, and a
last-token output projection. Every timed request is checked for correct logits
and exact greedy tokens. Incorrect outputs receive zero reward.

V2 times batch-one requests and requires small batch-one/two cache correctness
probes. The reset observation declares both requirements before the agent acts.
It also embeds the complete task, measurement and RL rules with SHA256 hashes;
the standalone wheel ships the same documents.

```bash
cd inference
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python assets.py
.venv/bin/python scorer.py solution.py --episode episodes/my-run --device cpu
```

`--device mps` selects the Mac GPU; MPS correctness and performance remain unverified.
Use `--diagnostic` while the machine is busy:
it verifies correctness but provides no timing reward. Read `record.json` or use
the [Python interface](interface.py); blocked/error records have null rewards.

- [Coding-agent reset/action/termination API and training split](RL-CONTRACT.md)
- [Installation and reproduction](REPRODUCE.md)
- [Independent versioned Python package](PACKAGING.md)
- [Recheck a saved run without inference](replay.py)
- [Task, numerical checks, reward, and resource budgets](TASK.md)
- [Measurement protocol and trust boundary](DESIGN.md)
- [Release validation and evidence scope](../VALIDATION.md)

This is a limited alpha for trusted research code. Real pretrained CPU terminal
workloads pass correctness; reliable standard speed rewards remain unverified.
Fresh local dependency installation and narrow framework boundary compatibility
have been checked separately. Coding-policy weight training, learning gains,
second-machine verification, hostile-code isolation and CUDA are unverified.
The other training and kernel environments are not needed to run this folder.
