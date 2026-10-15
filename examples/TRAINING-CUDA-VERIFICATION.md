# Training Improvement CUDA verification

Results recorded and independently reviewed on October 15, 2026.

An LLM supplies complete Python training code. `TrainingEnv.reset()` returns the
task and constraints; `TrainingEnv.step({"code": generated_source, "final": True})`
runs the task model, independently scores its loss, and returns a terminal
response. Your RL trainer consumes eligible standard training rewards. Successful
reward is `exp(-loss)`. Candidate-invalid zero rewards and infrastructure null
rewards remain distinct; smoke, nonterminal, and held-out results must not update
the outer policy.

Start with the [complete example](training-cuda-alpha4/README.md) and its
[Python API guide](training-cuda-alpha4/training-env/RL-INTEGRATION.md).

## Execution status

| Check | Observed result | Independent acceptance |
|---|---|---|
| Exact source archive and static checks | Passed. | Accepted. |
| Focused fixtures and runtime preflight | Passed. | Accepted. |
| Public data preparation, CUDA evaluator, and saved-weight audit | Completed successfully. | Accepted. |
| Native `TrainingEnv` reset/final interaction, seed 14 | Completed successfully with a native trainable dev reward. | Accepted. |
| Complete namespaced source-layout check | Passed without changing public repository files. | Accepted. |

The completed evaluator example used the supplied recipe. It did not generate
code with an LLM, consume a reward in an RL trainer, or update outer LLM weights.
The native API interaction is a separate check; evaluator completion does not
establish its result.

## Measured evaluator result

`training-env/public_cuda_example.py` verified its public raw-data cache,
prepared the corpus, calibrated the GPU, and ran the supplied learning-rate 0.001
recipe with seed 19001. A separate grader loaded the trained checkpoint and
scored dev predictions.

| Measurement | Recorded value |
|---|---:|
| Task model parameters | 5,063,424 |
| Training calls | 500 |
| Training tokens | 4,096,000 |
| Dev predictions graded | 491,520 |
| Development cross-entropy loss | 3.2106199502944945 |
| Native reward, `exp(-loss)` | 0.040331601931443396 |
| Changed state tensors | 65 |
| Complete example elapsed time, including preparation and checks | About 160.7 seconds |

The grader's loaded final tensor hash matched the trained checkpoint. The saved
weight audit reported consistent tensor metadata and checkpoint contents, with
identical initial/final state specifications and different weight values:

```text
Initial tensor SHA256:
78c9e30543a02e975146b1346f35a49ea8428a823e21401a6b8be8a1d6cc03cc

Final tensor SHA256, also loaded by the independent grader:
4c525f6c5351a73fd0934245b0634413fec00d75554e7009860703d159ce90d2
```

The recorded source and prepared data identities remained unchanged. Calibration
and cleanup checks passed, and the process supervisors reported no discovery or
cleanup errors. Independent review accepted the combined evaluator, weight,
calibration, supervision, and cleanup evidence for this exact run. The saved-weight
audit alone would not establish that complete result. Final-test data was checked
for integrity but was not loss-graded.
This dev result does not establish an improvement over a matched baseline.

## Native API result

The separate native check called `training-env/interaction.py reset` and then
`final` with `training-env/starter.py`, seed 14, standard mode, and the training
task split. Reset returned a fresh, unfinalized episode with one trial remaining.
The terminal response reported:

| Field | Recorded value |
|---|---:|
| Development cross-entropy loss | 3.23127859433492 |
| Native terminal reward | 0.03950695311513604 |
| `terminated` | `true` |
| `truncated` | `false` |
| `info.trainable` | `true` |
| `info.exclude_from_rl` | `false` |
| Scored data split | `dev` |
| Complete API check elapsed time | About 74.46 seconds |

It used the same 500-call task budget, model size, prepared data identity, and
source archive as the evaluator example. The native record reported changed
parameters and independent grading. The source, reward, calibration, and cleanup
checks passed. Independent review accepted this exact native standard/train/dev
reset/final compatibility result. The reward was not consumed by a trainer. This
API result is distinct from the initial/final saved-weight audit reported for the
evaluator example above.

## Observed hardware, runtime, and budgets

| Item | Observed configuration or fixed limit |
|---|---|
| Host | Linux x86_64 |
| GPU | One NVIDIA RTX A2000, 6,138 MiB reported memory |
| NVIDIA driver | 595.91.07 |
| Python | 3.12.14 |
| Torch | 2.11.0+cu128, CUDA 12.8 |
| NumPy | 2.5.3 |
| tokenizers | 0.22.1 |
| Task | TinyStories, vocabulary 4,096, context 256, batch 32 |
| Task limits | 5,100,000 parameters and 64,000,000 tensor bytes |
| Training time limit | 300 seconds |
| Example phase limits | Preparation 300 seconds; calibration 90 seconds; native episode 600 seconds; saved-weight audit 30 seconds |
| Complete example work limit | 1,200 seconds, with a 15-second internal cleanup reserve and a separate outer cleanup allowance of 20 seconds |
| Native API check limits | Seed 14, standard/train/dev, 1,050-second work limit and 20-second cleanup allowance |

There was no hard aggregate RAM or VRAM quota. These results cover the observed
runtime, not fresh dependency or driver installation. Execution on other GPUs
remains unverified. The supplied model and public data preparation define this
task; arbitrary replacement workloads and outer LLM learning gains are not
established by this check.

## Exact source download

The tested standalone ZIP has 53 source files and is 129,146 bytes. Its contents
are the same files published under `examples/training-cuda-alpha4/`, without that
directory prefix inside the ZIP.

```text
Source ZIP SHA256:
22ccd136f8bb43a2363ad1a5d532e194ed53b649ac09df360cead92d8459bf42

Accepted source manifest SHA256:
58de53e9dc4820527f2d1901019c051cab1a076dfd37bad623bd33739352fb74
```

The tested bundle is immutable. Its README retains the wording written before
execution; this dated page records subsequent results and their review status.
