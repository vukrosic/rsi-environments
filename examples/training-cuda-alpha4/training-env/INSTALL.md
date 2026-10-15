# Source checkout installation

Extract the source archive into a new directory. Keep `training-env/` and
`suite/` as siblings and use the extracted root as your working directory.
The checkout is the installation; no training wheel or RL framework is needed.

Use an existing compatible Linux NVIDIA RTX A2000 environment:
[observed requirements](requirements-cuda-observed.txt) pin Python 3.12.14,
Torch 2.11.0+cu128, NumPy 2.5.3 and tokenizers 0.22.1. The driver must already
support this runtime. Fresh dependency/driver installation is not verified here.
The example does not install packages or download model weights.

Set `CUDA_PYTHON` and `TRAINING_PYTHON` to the same absolute intended interpreter.
If dependencies are in an existing target directory, set `PYTHONPATH` to that
explicit overlay for the native commands and their subprocesses; set
`PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1`, and unset `PYTHONHOME`.
Use the public one-command example's repeatable `--overlay` for that command.
Do not use `-E` with the native commands when relying on this explicit PYTHONPATH.
Do not replace host Torch/NumPy or inherit unrelated Python paths.

[RAW-DATA.md](RAW-DATA.md) prepares a new corpus from bounded pinned official
TinyStories text or a reverified cache. It does not reconstruct historical
private prepared bytes or losses. Keep the cache, prepared data, calibration,
job outputs and checkpoints outside this source checkout. No data is bundled.
The source export requires both `raw-data-profile.json` and
`public_cuda_example-spec.json`; a missing helper or JSON blocks execution.

Follow [the root README](../README.md) for a native reset/final command and
[PUBLIC-CUDA.md](PUBLIC-CUDA.md) for the separate preparation/training/audit
command. Both require idle-device, source/data, supervision and cleanup proof;
source preparation and metadata preflight are not accepted training results.
