# Alpha4: Training Improvement

An LLM submits training code. The environment runs the task model, independently
scores its prediction loss, and returns an eligible reward to your RL trainer.
This release supports experiments on training improvement. You supply the outer
LLM, source generation, and RL trainer.

Extract the source checkout, keep `training-env/` and `suite/` as siblings, and
follow the [quick start](README.md). It uses `training-env/calibrate.py` and the
`reset` and `final` commands in `training-env/interaction.py` to submit
`training-env/starter.py`. To submit LLM-generated source, call `TrainingEnv.reset()`
and then `TrainingEnv.step({"code": generated_source, "final": True})`.
See the [Python API guide](training-env/RL-INTEGRATION.md).

Your trainer should consume only eligible standard terminal training rewards.
Successful reward equals `exp(-loss)`. Preserve candidate-invalid zero rewards,
exclude infrastructure null rewards, and exclude smoke, nonterminal, and held-out
results from policy updates.

The supplied task is a 5,063,424-parameter GPT trained on TinyStories text for
500 calls. The observed runtime is Linux with an NVIDIA RTX A2000, Python 3.12.14,
Torch 2.11.0+cu128, NumPy 2.5.3, and tokenizers 0.22.1. A training wheel is
unnecessary. Fresh dependency installation and execution on other GPUs remain
unverified.

`training-env/prepare_public_data.py` prepares fresh public data.
`training-env/public_cuda_example.py` runs a separate preparation, training,
evaluation, and saved-weight audit command. See
[PUBLIC-CUDA.md](training-env/PUBLIC-CUDA.md). That command and the native API
interaction still await end-to-end verification for this release. We have not
trained your outer LLM or demonstrated learning gains.
