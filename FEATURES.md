# Features

An LLM submits code. The environment runs it within a fixed budget, measures the result and returns feedback and a reward. Your RL trainer uses that reward to update the LLM.

You provide the LLM and trainer. We provide the submission interface, execution controls, grader and reward records. Task models and datasets currently follow each environment's configuration; arbitrary replacements are not yet supported.

Updated October 15, 2026. The latest release is [v0.1.0-alpha.4](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.4). It adds the standalone Training Improvement CUDA example; other NVIDIA development results below are not included in that release.

## Environments

| Environment | What the LLM changes | How it is scored | Current status |
|---|---|---|---|
| [Inference optimization](inference/README.md) | Model inference code | Latency, after output correctness checks | Main fixes evaluator error classification; seven mocked API checks passed on Linux. Earlier tags retain the defect. Timing rewards remain unqualified; this repair adds no CUDA support. A development NVIDIA comparison had no verified speedup. |
| [Training improvement](examples/training-cuda-alpha4/README.md) | Model, optimizer or training-step code | Prediction loss within the training budget | Alpha4 includes two verified 500-call NVIDIA runs, including a terminal native API reward. Saved-weight changes were audited in the example run. Fresh dependency installation remains unverified. |
| [Kernel optimization](kernel-env/README.md) | Numerical operations used in neural networks | Runtime, after numerical accuracy checks | Published tasks and API. Current v1 timing has unresolved deferred-output and predictable-input issues; speed rewards are unverified for RL. NVIDIA diagnostics ran, but timing also exceeded the stability threshold. |
| Model compression | Weight representation or model execution | Runtime allocation, subject to a fixed quality limit | Not published yet. Apple and NVIDIA measurements ran; one NVIDIA candidate became the next incumbent. Independent reference setup and trainer integration are unfinished. |

## Connecting a trainer

1. Sample code from your LLM.
2. Submit it through the environment's Python API or the shared Verifiers adapter.
3. Give the feedback to the LLM for another attempt, or make a terminal submission.
4. Pass eligible terminal rewards to your RL trainer.

Training improvement returns `exp(-loss)` for valid terminal submissions. Other objectives have their own scoring rules. Failed infrastructure, unreliable timing, smoke checks and evaluation-only episodes must be excluded from policy updates.

Start with the [alpha4 training example](examples/training-cuda-alpha4/README.md) and its [dated CUDA verification](examples/TRAINING-CUDA-VERIFICATION.md). The [Verifiers integration guide](suite/integration/README.md) covers earlier framework compatibility checks; the alpha4 CUDA run used the native API. The reward limitations above apply. Published examples do not include an outer LLM trainer or establish learning gains or recursive self-improvement.

## Execution and results

Records contain source identities, seeds, metrics and reward eligibility. Large models, datasets and checkpoints are prepared separately and kept out of Git. Compression allocation measurements describe PyTorch or MLX tracked memory, not total physical VRAM.

Submitted Python runs with the user's permissions. The process controls are not a security sandbox; untrusted submissions require isolated execution.

[Release validation](VALIDATION.md) · [Roadmap](ROADMAP.md) · [README](README.md)
