# Kernel optimization RL environment — release candidate 0.1.1

An environment for an external RL trainer to teach a coding agent to optimize
six float32 training kernels. The policy produces Python code, the environment
executes a frozen snapshot and returns a correctness-gated terminal reward, and
the external trainer updates the policy's weights. This repository supplies the
environment side. No coding-agent weight training or learning gain is claimed.

The four training tasks are `rmsnorm`, `softmax`, `swiglu`, and `cross_entropy`.
The two held-out evaluation tasks are `attention` and `newton_schulz`. These are
different operations, not merely new random seeds. Development and terminal
scoring also use distinct seeds and shapes. All cases are public, reproducible
synthetic kernel inputs; they are not secret tests or a malicious-code defense.
Use real datasets/weights for model tasks; numerical kernel tasks have no
dataset download or pretrained-model dependency.

## Install and reproduce on CPU

Python 3.11+ on macOS or Linux. Install a suitable PyTorch build for your device.
The verified local runtime is Python 3.12.8, PyTorch 2.8.0 and NumPy 2.2.5 on
Apple M4; see the actual hardware/version values in the example record.

```sh
cd kernel-env
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[test]'
kernel-env init softmax work/candidate.py
kernel-env evaluate softmax work/candidate.py --profile smoke --device cpu --seed 17
python -m pytest -q
```

The CLI works without installing from this directory as
`python -m kernel_env.cli`. Smoke workloads are small development
checks. Their numeric reward is useful to test integration and must not be
mixed into standard research results.

## Environment-side RL API

```python
from kernel_env import KernelEnvironment

env = KernelEnvironment("softmax", "work/rollout", split="train",
                        profile="smoke", device="cpu", seed=17,
                        max_steps=3, max_episode_seconds=120)
observation = env.reset()
# A model receives observation["instruction"], observation["code"], and feedback.
transition = env.step({"type": "test", "code": observation["code"]})
assert transition["reward"] == 0.0  # test result is feedback, not episode reward
transition = env.step({"type": "submit"})
assert transition["terminated"] and not transition["truncated"]
terminal_reward = transition["reward"]
record = transition["info"]  # suite v1 record, raw timing and source provenance
# An external trainer uses this reward to update model weights.
```

Actions are JSON dictionaries: `edit` requires a `code` string; `test` and
`submit` optionally supply replacement code. `edit` writes editable code;
`test` runs development cases and returns tool feedback with zero intermediate
reward; `submit` freezes the current code and scores terminal cases. The last
budgeted action automatically submits. A string action is shorthand for
`{"type": "test", "code": string}`. `final()` is shorthand for submit.

`reset()` starts a unique episode directory and preserves previous evidence.
`step()` returns `observation`, `reward`, `terminated`, `truncated`, and `info`.
Successful or incorrect terminal submissions set `terminated=True`; invalid
outputs earn zero. Infrastructure blocks and execution failures set
`truncated=True` and `reward=None`. A trainer must exclude those episodes from
reward optimization, retaining them as infrastructure diagnostics. Further
actions require another reset. Malformed action schemas raise `ValueError`;
the framework adapter must convert or reject malformed policy text.

Both transitions and score records contain `task_split`, `trainable`,
`exclude_from_rl`, and `rl_exclusion_reasons`. Evaluation-only tasks and null
infrastructure failures are always excluded from policy updates. Only terminal
standard training scores are eligible, including invalid training submissions
with numeric reward 0. Smoke and intermediate feedback stay excluded. A numeric
evaluation score is for reporting and never overrides the exclusion marker.
Every step saves `transition-NN.json` with these markers in its episode folder.

Observations include a `task_spec` describing the exact operation, signature,
constants, nominal shapes/dtypes, and error tolerance. Agents can understand
the task without reading the scorer; implementations must still support the
documented shape variants.

See [RL_API.md](RL_API.md) for precise integration requirements. There is no
duplicate trainer framework here: the coordinating chat owns the shared,
version-pinned framework adapter. Local rollout compatibility tests exercise
these transitions; they do not establish a model-training result.

## Standard timing on Apple MPS

```sh
kernel-env calibrate softmax --device mps --output-dir work/calibrations
# Use the successful receipt's printed path. Do not use a blocked receipt.
kernel-env evaluate softmax work/candidate.py --profile standard --device mps \
  --phase final --calibration work/calibrations/calibration-ID/calibration.json \
  --timeout 300
```

All GPU activity acquires `suite.resources.accelerator_lock`; the default
shared lease is `~/.cache/vuk-research-envs/accelerator.lock`. The packaged
`suite/` files are verbatim licensed copies of the coordinating suite helpers,
so this folder can be installed independently. CPU runs do not acquire a GPU
lease. Lock waits and worker communication are bounded by the evaluation's
wall-clock deadline.

The lock is cooperative. Older Claude experiments and unrelated apps may not
use it. Recognized games/renders/legacy kernel benchmarks defer GPU evaluation.
Calibration also rejects unrelated processes using more than 50% CPU. These
checks are conservative and incomplete, not proof of GPU idleness. No process
outside the evaluator's own subprocess groups is terminated.

Calibration records hardware, dependencies, scorer/worker hashes, observed
load, and three baseline samples per development/final shape variant. Within
each variant, a spread above 10% blocks calibration. Standard timing requires
that fixed receipt, matching installation/system/task/scorer hashes. A baseline
more than 10% slower than idle, or a bracketing ratio outside 0.90–1.10, blocks
the reward. A new receipt cannot rescue a loaded run: calibrate on an inspected
idle machine. Final scoring uses exactly five measured trials and a median
paired speedup. Small gaps inside the observed noise are not optimization
findings. Raw samples remain available for analysis.

CUDA has conditional synchronization, device handling, and TF32-disabled
matmul paths; CUDA is unverified on this Mac. NVIDIA hardware is not required.
The process-group/resource helpers use POSIX APIs, so Windows is not supported
by this release.

## What is timed and checked

Each call receives fresh local-generator CPU inputs, transferred before timing.
Candidate and baseline are separate subprocesses using the same protocol. The
parent measures the `run` command through the device-synchronized `done`
message. This includes small IPC/scheduling overhead on both sides, while
input generation, transfer, output transfer, and checks are outside the timer.
It is a reproducible execution latency metric, not an isolated GPU event time.
Tiny kernels can be dominated by protocol overhead; smoke speedups are not
research evidence. Compilation and initialization in the warmups are excluded;
timed candidate recompilation is included.

Every candidate output, including unscored zero/large/tiny numerical stress
fixtures and two warmups, is checked against independent CPU mathematics in
the parent. All scored cases contain fresh random values. The timed built-in
baseline is bracketed around the candidate with fresh values on its second
call. Wrong answers never receive speed reward. Checks require float32 finite
outputs of the exact shape on the requested device and unchanged inputs.
The error is `max(abs(actual-expected)/max(1,abs(expected)))`: tolerance `2e-3`
for five tasks and `2e-2` for the fixed five-step Newton–Schulz iteration.

| Task | Standard nominal input | Competitive baseline |
| --- | --- | --- |
| rmsnorm | 8192 × 1024 plus 1024 weights | `F.rms_norm`, epsilon 1e-6 |
| softmax | 4096 × 4096 | `torch.softmax` on last axis |
| swiglu | two 8192 × 2816 matrices | `F.silu(a) * b` |
| cross_entropy | 8192 × 4096 logits, 8192 int64 labels | `F.cross_entropy`, mean reduction |
| attention | Q/K/V each 8 × 8 × 1024 × 64 | causal `F.scaled_dot_product_attention` |
| newton_schulz | ten 704 × 256 matrices | five iterations of batched PyTorch matmul |

Input variants change dimensions and numerical distributions. Final variants
use dimensions absent from development feedback. Optimized built-ins, custom
Metal shaders, and other single-file methods are permitted if they satisfy the
same correctness/resource contract. The baseline is already optimized; copying
it is a valid control and does not demonstrate an optimization improvement.

## Evidence and boundaries

`evaluate(solution, task, ...)` returns a normalized suite v1 score record:
`ok` reward is `speedup/(1+speedup)`; `invalid` reward is 0; `blocked`/`error`
rewards are null. Every invocation saves a content-hashed read-only candidate
snapshot, config, current/failed case, partial trial evidence, worker logs,
and `record.json` in a unique directory. Read-only files are an accidental-edit
guard; they are not immutable against their owning user. Candidate and trusted
source hashes are checked before a speed reward is returned. A failed trusted
baseline blocks the score rather than penalizing the candidate.

Submitted code runs only in a separate bounded process group; outputs use
bytes, JSON, and NPZ with `allow_pickle=False`, never candidate-controlled
pickle objects in the parent. The parent checks outputs and computes reward.
This improves accidental fault containment and catches ordinary cached outputs
or scorer-file replacement. It is **not a security sandbox**. Candidates have
the user's filesystem/network permissions, can inspect the public scorer and
protocol, and can tamper with their own runner, mutation metadata, or timing
messages. Archive decompression, total memory/disk/network usage and escaping
process groups are not hardened. A malicious agent can still manipulate timing
or attack files. Use a separate unprivileged container/VM and protected scorer
service before training with hostile code. Do not represent this release as
secure against reward hacking.

Historical `rsi/kernels/` and the earlier `research-envs` adapter supplied
operator definitions, workload dimensions and useful baseline pieces. They
were read as references and remain unchanged by this environment build.
Fresh-input timing, independent parent checks, episode control and evidence
collection are implemented and tested here. See [release validation](../VALIDATION.md)
for actual release checks and remaining GPU/trainer limitations.

Softmax correctness additionally requires probabilities in `[0, 1]` and each row
to sum to one within the task tolerance. This prevents zero or missing-mass
outputs from passing only because each probability is individually small.
