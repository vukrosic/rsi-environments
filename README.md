# RSI Environments

**Reinforcement learning environments for LLMs that improve training and inference code.**

An LLM submits code. The environment executes it, measures the result and returns feedback and a reward. You connect those rewards to your own RL trainer.

[Start here](docs/GETTING-STARTED.md) · [Features and status](FEATURES.md) · [Roadmap](ROADMAP.md) · [Releases](https://github.com/vukrosic/rsi-environments/releases)

## Environments

| Environment | What the LLM changes | Objective |
|---|---|---|
| [Inference optimization](inference/README.md) | Model inference code | Lower latency while preserving outputs |
| [Training improvement](training-env/README.md) | Model, optimizer or training step | Lower prediction loss within the task budget |
| [Kernel optimization](kernel-env/README.md) | Numerical operations | Lower runtime while preserving accuracy |

**Arenas (in development):** 37 smaller CPU environments (37 with tests passing) where each reward is measured against the current best submission for that task, never a fixed target. The code is not released yet; [the arena list, spec and backlog](arenas/README.md) are public.

Model compression is in development and is not included in the current alpha. It measures runtime allocation subject to a fixed model-quality limit.

The model trained or evaluated inside a task is separate from the LLM your RL trainer updates. Each environment currently specifies its task model and data; configuring arbitrary replacements is planned.

## Release status

The latest release, [v0.1.0-alpha.4](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.4), adds a standalone Training Improvement CUDA example. Two NVIDIA runs completed 500 training calls each; the native Python API returned a terminal training reward. Fresh dependency installation is still being verified. Earlier releases include the three environment APIs, a shared Verifiers adapter and a bounded Apple GPU training check.

**Inference API fix — October 15, 2026:** current `main` excludes exceptions escaping the evaluator from RL updates, rather than treating them as invalid submitted code with zero reward. Seven mocked-evaluator checks passed on the Linux server. Earlier release tags still contain the defect; use current `main` for this repair. These checks do not qualify inference timing rewards or add CUDA support. The alpha4 Training Improvement example uses a separate implementation and is unaffected. See [verification details](inference/API-ERROR-VERIFICATION.json).

Kernel speed rewards have unresolved timing-boundary and predictable-input issues; do not use them for RL updates until fixed. Standard CUDA timing also failed the stability threshold on the tested NVIDIA host. Compression is not included in this release.

See [Features](FEATURES.md) for the development status and [VALIDATION.md](VALIDATION.md) for checks of the published code. The published examples do not include an outer LLM trainer or establish learning gains or recursive self-improvement.

**Kernel benchmark advisory — October 16:** current v5 has predictable timed
inputs and an incomplete fingerprint-length check. Its speedups remain excluded
from RL updates pending repairs and validation. See [the known limits](kernel-timer/KNOWN-LIMITATIONS.md).

| Benchmark | Published checks |
|---|---|
| [Corrected NVIDIA kernel benchmark](kernel-timer/) | Library baselines and matched timing; 16 known fakes scored no gain on RTX A2000. The first 36-step GRPO pilot showed no measured improvement. |

The standalone v3 benchmark and selected GPU examples report development measurements; their speedups are not qualified for RL updates.

[Triton cross-entropy example](examples/gpu-cross-entropy/): exact submitted code,
workload and recorded result. One selected retime reported 2.231× the PyTorch
library baseline's speed; the example explains its measurement and provenance limits.

[Causal attention example](examples/gpu-attention/): exact submitted code and a
selected 3.879× retime against PyTorch SDPA. It uses float16 dot products with
float32 inputs/output; its measured error passes the documented task tolerance.

[Kernel SFT seed and actual results](experiments/kernel-sft-seed/): 151 paired training examples; more code replies, with no demonstrated correctness improvement.

## Training Improvement CUDA example

An LLM submits training code. The environment runs the task model, independently
scores its prediction loss, and returns an eligible reward to your RL trainer.
The complete [Training Improvement checkout](examples/training-cuda-alpha4/README.md)
contains the public-data preparation command, native reset/final API, and CUDA
training and saved-weight audit example. You supply the outer LLM and trainer.

From this repository, start with:

```sh
cd examples/training-cuda-alpha4
```

Follow that directory's README for the actual commands and compatibility limits.
See the [dated verification page](examples/TRAINING-CUDA-VERIFICATION.md) for execution results and review status. The
[alpha4 notes](examples/training-cuda-alpha4/ALPHA4.md) explain source submissions
and eligible rewards. This standalone example has its own source manifest; it
does not replace the other environments or the repository-wide manifest.

## Installation check

Use Python **3.12** for the complete suite. The kernel package also supports Python 3.11+. This small CPU example checks installation and interaction:

```bash
git clone https://github.com/vukrosic/rsi-environments.git
cd rsi-environments
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './kernel-env[test]'

cd kernel-env
kernel-env init softmax work/softmax.py
kernel-env evaluate softmax work/softmax.py --profile smoke --device cpu --trials 3
```

Smoke results are excluded from RL training rewards. Follow the [inference setup](inference/README.md), [training setup](training-env/INSTALL.md) or [kernel setup](kernel-env/README.md) for each task. Model weights and datasets are prepared separately.

## Python interface

From the installed kernel package:

```python
from kernel_env import KernelEnvironment

env = KernelEnvironment(
    "softmax", "work/episode", split="train",
    profile="smoke", device="cpu", seed=17, max_steps=2,
)
observation = env.reset()

# Supply source generated by your LLM instead of the starter.
transition = env.step({"type": "test", "code": observation["code"]})
transition = env.step({"type": "submit"})

print(transition["reward"], transition["terminated"])
print(transition["info"]["exclude_from_rl"])
```

Native APIs differ. The [shared adapter](suite/integration/README.md) translates them into one Verifiers interface. The [training contract](training-env/RL-INTEGRATION.md) explains code submissions, feedback and terminal rewards.

## RL integration

The adapter is pinned to **Prime Intellect Verifiers 0.4.0**. Compatibility checks exercised framework loading, code submissions, feedback and terminal results across the three CPU environments. The compatibility provider returns starter code; it does not train an LLM.

After installing the native task dependencies:

```bash
python -m pip install -r suite/integration/requirements-framework.lock
python -m pip install -r suite/integration/requirements-runtime.txt
python -m pip install --no-deps -e suite/integration
```

You supply the LLM, sampling, optimizer and checkpoints. Use only eligible terminal training rewards. Incorrect code receives zero; infrastructure failures, unreliable timing, smoke checks and evaluation-only results are excluded. See the [integration guide](suite/integration/README.md) for configuration and exclusion handling.

## Execution limits

Submitted Python runs with the user's permissions. Process and resource controls are not a security sandbox. Use isolated execution for untrusted code.

Participating GPU workers share a cooperative accelerator lock. Other applications do not honor it, so timing still requires an idle device and valid calibration. Hardware support and verification differ by environment.

Source identities, seeds, raw metrics and eligibility reasons are recorded with results. Large models, datasets and checkpoints are not stored in Git; downloaded assets retain their own licenses.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md) and [the roadmap](ROADMAP.md).

Created by **Vuk Rosić**. MIT licensed.
