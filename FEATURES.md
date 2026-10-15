# Features and current status

Reinforcement learning environments for improving ML code: an LLM proposes code, the environment executes it under a task budget, and an independent grader returns feedback and a reward. An external RL trainer can use eligible terminal rewards to update the coding LLM.

The research direction is recursive self-improvement. This project has not demonstrated recursive gains or trained a coding policy to improve itself.

Updated October 15, 2026. [Published release: v0.1.0-alpha.3](https://github.com/vukrosic/rsi-environments/releases/tag/v0.1.0-alpha.3). The development work described below is not yet included in that release.

## What the environment provides

- A programming task, starter implementation and submission interface.
- Bounded candidate execution and feedback for iterative code changes.
- Independent correctness or quality evaluation, resource limits and scoring.
- Terminal reward records with source identities, raw metrics and eligibility information.
- Separate training and evaluation partitions; smoke, infrastructure failures and unreliable measurements are excluded from policy updates.
- Native Python interfaces and a shared adapter pinned to Verifiers 0.4.0.

You provide your coding LLM and RL trainer, including policy sampling, optimization and checkpoints. The small model trained or evaluated inside a task is separate from the coding LLM your trainer updates. The repository supplies environments and integration examples; it does not supply a verified end-to-end coding-policy training system.

Current tasks have specified workloads and asset preparation. Using your own coding LLM does not mean arbitrary task models and datasets are already supported. Configurable task workloads are on the [roadmap](ROADMAP.md).

## Environments

| Environment | Agent's action | Measured objective | Published alpha | Development work for the next release |
|---|---|---|---|---|
| [Inference optimization](inference/README.md) | Rewrite a model's inference implementation | Reduce latency while preserving required outputs | Native task API, output checks and shared adapter | Real pretrained CUDA comparisons completed; the tested candidate had no verified speedup. Portable packet creation and terminal reward checks are being verified. |
| [Training improvement](training-env/README.md) | Change model, optimizer or training-step code | Reduce prediction loss within the task budget | Code submission, independent loss grading, Python API and shared adapter; bounded Apple GPU execution checked | NVIDIA training and a matched recipe comparison completed with saved-weight checks. The tested recipe change made loss worse. Portable reproduction and packaging are being verified. |
| [Kernel optimization](kernel-env/README.md) | Rewrite numerical operations used in neural networks | Reduce execution time while preserving numerical accuracy | Kernel tasks, correctness checks, CLI, Python API and shared adapter | NVIDIA correctness and diagnostic execution completed. Timing failed the stability gate, so standard CUDA speed rewards remain unqualified on the tested host. |
| Model compression | Change a language model's weight representation or execution | Reduce measured runtime allocation while meeting a fixed original-model quality limit | Not included in alpha.3 | Native Apple and NVIDIA measurements completed; an NVIDIA competition accepted one candidate and advanced its incumbent. Portable reference generation, external trainer integration and public source release are unfinished. |

Development execution is not a promise that the current public checkout reproduces those NVIDIA results. The next release must include matching source, dependency specifications, commands and compact evidence together. See the [release validation](VALIDATION.md) for checks of the published code.

## Rewards researchers can connect to a trainer

The interaction is: sample candidate code, submit it, read feedback, make a terminal submission, and pass an eligible reward to your trainer. Native APIs differ; the shared adapter translates them into one framework interface.

Training improvement uses independently measured loss: valid terminal reward is `exp(-loss)`. Inference and kernel objectives require valid numerical checks and timing conditions. A noisy or blocked run is not a slow candidate and must not become a training label. Compression's development objective measures PyTorch or MLX tracked allocation; those metrics are not total physical VRAM.

Start with the [training environment contract](training-env/RL-INTEGRATION.md) or the [shared integration guide](suite/integration/README.md). Actual GPU terminal reward integration with an external framework is still being validated. No coding-policy weight update or learning gain is claimed.

## Execution limits

GPU support and evidence are specific to each environment, backend and workload. A successful GPU model run does not certify timing rewards or trainer compatibility. Fixed budgets and independently measured quality remain part of the task contract; submissions cannot declare their own score.

Submitted code runs with the local user's permissions. Process and resource controls are not a hardened sandbox. Use appropriately isolated execution for untrusted code.

Models, datasets, checkpoints and large run artifacts are not bundled in Git. Follow the asset preparation and licensing instructions for each task.

[Roadmap](ROADMAP.md) · [README](README.md) · [Releases](https://github.com/vukrosic/rsi-environments/releases)
