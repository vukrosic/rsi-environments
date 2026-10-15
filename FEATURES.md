# Features

An LLM submits code. The environment runs it within a fixed budget, measures the result and returns feedback and a reward. Your RL trainer uses that reward to update the LLM.

You provide the LLM and trainer. We provide the submission interface, execution controls, grader and reward records. Task models and datasets currently follow each environment's configuration; arbitrary replacements are not yet supported.

Updated October 15, 2026. The published version is [v0.1.0-alpha.3](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.3). NVIDIA development results below are not yet included in that release.

## Environments

| Environment | What the LLM changes | How it is scored | Current status |
|---|---|---|---|
| [Inference optimization](inference/README.md) | Model inference code | Latency, after output correctness checks | Published API. NVIDIA comparisons ran in development; the tested change had no verified speedup. Portable setup and terminal reward checks are unfinished. |
| [Training improvement](training-env/README.md) | Model, optimizer or training-step code | Prediction loss within the training budget | Published API and bounded Apple GPU check. NVIDIA training and a recipe comparison ran in development. The tested recipe increased loss. Portable reproduction is unfinished. |
| [Kernel optimization](kernel-env/README.md) | Numerical operations used in neural networks | Runtime, after numerical accuracy checks | Published tasks and API. NVIDIA diagnostics ran, but timing exceeded the stability threshold. Standard CUDA speed rewards remain unavailable on the tested host. |
| Model compression | Weight representation or model execution | Runtime allocation, subject to a fixed quality limit | Not published yet. Apple and NVIDIA measurements ran; one NVIDIA candidate became the next incumbent. Independent reference setup and trainer integration are unfinished. |

## Connecting a trainer

1. Sample code from your LLM.
2. Submit it through the environment's Python API or the shared Verifiers adapter.
3. Give the feedback to the LLM for another attempt, or make a terminal submission.
4. Pass eligible terminal rewards to your RL trainer.

Training improvement returns `exp(-loss)` for valid terminal submissions. Other objectives have their own scoring rules. Failed infrastructure, unreliable timing, smoke checks and evaluation-only episodes must be excluded from policy updates.

See the [training API](training-env/RL-INTEGRATION.md) and [Verifiers integration guide](suite/integration/README.md). Framework compatibility checks have run; GPU terminal reward integration is still being validated. We have not demonstrated RL weight updates to an LLM or recursive self-improvement.

## Execution and results

Records contain source identities, seeds, metrics and reward eligibility. Large models, datasets and checkpoints are prepared separately and kept out of Git. Compression allocation measurements describe PyTorch or MLX tracked memory, not total physical VRAM.

Submitted Python runs with the user's permissions. The process controls are not a security sandbox; untrusted submissions require isolated execution.

[Release validation](VALIDATION.md) · [Roadmap](ROADMAP.md) · [README](README.md)
