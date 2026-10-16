# Validation and evidence scope

## Kernel timer v5 checks — October 16, 2026

Returned RTX A2000 outputs report a passing gate with 12 unchanged rows and no
listed problems. Ten process-identity and cleanup checks passed. The attention
adversarial check ran 20 cases with scanning enabled and 20 with it disabled;
the runner reported no gains above its threshold in either mode. These are
checks of the tested cases, rather than exhaustive isolation or timing proof.

The published `kernel-timer/bench3.py` hash matches the source hash supplied by
the operator. Four returned output hashes and the precise evidence limits are
in [the v5 verification record](kernel-timer/VALIDATION-V5.json). Individual
unchanged-case rows and the check-runner source were not returned in this
evidence set; output integrity does not independently attest the full remote
runtime. These checks add no LLM learning claim and do not qualify speed rewards
from the older kernel environment API.

## Inference API exception repair on main — October 15, 2026

Current `main` separates action validation from evaluator execution. Exceptions
escaping the evaluator end the episode with null reward and `trainable=False`;
invalid submitted actions retain zero reward. Configuration is checked before
episode creation. Seven mocked-evaluator checks passed in 4.159 seconds on the
Linux server against the exact public dependency closure. Source and returned
results were independently reviewed. Python runtime and test bytes match that
execution; accompanying documentation now records the result.

The check used no model forward pass, GPU execution or timing benchmark. Module
origins were inspected in a separate fresh process under the admitted environment,
not captured inside the test process. See the [sanitized verification record](inference/API-ERROR-VERIFICATION.json).
Earlier release tags retain the defect that could misclassify an evaluator
exception as invalid submitted code with trainable zero reward. Use current
`main` for the repair; this does not qualify native inference speed rewards or
add CUDA support.

The alpha4 Training Improvement example uses a separate implementation and is
unaffected. The kernel reward limitations described in [Features](FEATURES.md)
also remain open.

## v0.1.0-alpha.4 — October 15, 2026

The standalone [Training Improvement example](examples/training-cuda-alpha4/README.md)
completed two NVIDIA CUDA runs with 500 training calls each. The example run
recorded changed saved weights and an independent loss audit. A separate native
`TrainingEnv` reset/final episode returned a terminal, training-eligible reward.
Source, controls, cleanup and actual results received independent review.

See the [dated verification page](examples/TRAINING-CUDA-VERIFICATION.md) for
runtime versions, losses, rewards, source hashes and exact evidence limits.
Fresh dependency installation, outer LLM weight updates, held-out learning gains
and outside adoption remain unverified. Historical sections below describe
their original releases; alpha4 does not replace the other environment APIs.

## v0.1.0-alpha.3 — October 14, 2026

One real TinyStories inner-GPT smoke completed on Apple MPS under the public
`training-v1.1-fixed-steps` smoke protocol. The run had 5,063,424 unique model
parameters, 3 scheduled training calls (2 with positive learning rate), 96
training tokens and 64 development grading tokens. The observed development
loss was 8.111218929290771; this is a single terminal loss, with no initial-loss
comparison or learning-gain claim. The smoke reward is excluded from RL
training.

An independent source/checkpoint/receipt review checked the observation
wrappers, original starter function behavior, MPS tensor boundaries, finite
state and initial/final weight hashes. That review establishes execution at
observed MPS tensor boundaries; CPU metadata, preprocessing, storage,
synchronization and instrumentation also participate. The public reproduction
command in [docs/apple-gpu-check.md](docs/apple-gpu-check.md) repeats the
unchanged starter workload without the private observation wrappers, so it does
not reproduce the full hash instrumentation or device-boundary receipt.

| Environment | Apple MPS evidence | CUDA scope |
|---|---|---|
| Inference optimization | Existing supported runtime is macOS CPU/Apple MPS | Unsupported |
| Training improvement | One real-data smoke completed on MPS | Not executed; compatibility not established |
| Kernel optimization | Apple/Metal paths exist for selected tasks | Conditional on backend and task; not executed in this release |

No before/after inner-model learning gain, standard-budget accelerator run,
standard CUDA execution, coding-agent weight update, learned policy gain, or
recursive improvement is established. See the sanitized machine-readable
[Apple GPU evidence](docs/apple-gpu-check.json). Historical alpha2 and alpha1
checks below retain their original scope. The [NVIDIA guide](docs/nvidia-training.md)
and example configurations are prepared and Mac schema/API/admission validated;
Linux CUDA execution and the full learner runtime are pending.

## v0.1.0-alpha.2 — October 14, 2026

This is a limited CPU source alpha. Native source fixes were captured in an
immutable independent review snapshot; the public export is checked separately.
The current artifact's exact file hashes are in `SOURCE-MANIFEST.json`.

| Evidence | Verified scope |
|---|---|
| Exported Verifiers adapter | 40 checks passed with Verifiers 0.4.0 |
| Exported shared helpers | 14 checks passed; standalone training import passed in an isolated dependency prefix |
| Exported kernel framework episode | Two actual scripted test/submit calls, correct CPU native result and terminal feedback |
| Reward exclusions | That smoke episode was deliberately rejected for RL scoring; no policy update |
| Native inference package | Final 0.2.3 wheel payload and installed modules checked; earlier real CPU terminal correctness predates the final thread guard |
| Native training dependencies | Isolated Python 3.12/PyTorch 2.8.0/NumPy 2.2.5/tokenizers 0.22.1 prefix checked on this Mac |
| Native training diagnostic | Saved initial/final weights, hashes and paired loss replay independently checked |
| Native kernel source package | Exact source archive independently rebuilt; installed runtime modules compared separately |

The versioned `cpu-training-v1` diagnostic trains a real 5,063,424-parameter GPT
on TinyStories for 20 optimizer calls / 5,120 training tokens. Seed17 paired
cross-entropy fell from approximately 8.3575 to 6.3965 on the fixed 1,024-token
grading samples. This is one diagnostic run: the inner model changed, while the
outer coding agent did not. Smoke and diagnostic scores cannot enter standard
RL updates. Dataset/weight/checkpoint files are deliberately excluded from Git;
follow training-env/INSTALL.md and PROTOCOL.md to prepare and reproduce tasks.

These native installation and diagnostic receipts concern independently frozen
builder artifacts. They are not a claim that every historical native test or
model experiment was rerun across the transformed public source. This export
retains the same accepted runtime hashes and checks its composition, imports
and framework episode separately. Package metadata and aggregate release
versions are distinct. Full second-machine installation remains unverified.

Run native suites separately because flat development module names can collide:

```bash
python -m pytest suite/tests -q
(cd inference && python -m pytest tests -q)
(cd training-env && python -m pytest tests -q)
(cd kernel-env && python -m pytest tests -q)
python -m pytest suite/integration/tests -q
```

The framework compatibility provider returns starter code. A trace can report
`ok=False` when correct native smoke execution deliberately receives an RL
scoring exclusion. This checks integration and exclusions, not policy training.
Standard speed rewards still require valid workload-specific calibration and
host-load checks. No calibrated GPU speedup, coding-agent weight update,
held-out policy learning gain, recursive improvement or external adoption is
claimed. Process ownership and cleanup are fault controls, not hostile-code
containment guarantees. Use trusted local Python.

## Historical v0.1.0-alpha.1 checks

The following counts belong to immutable alpha1 commit
`f4cfd5abf1887cd9c3e0d317b29deb764b99715e`; they are not fresh alpha2 results.
Python 3.12.8, PyTorch 2.8.0 and NumPy 2.2.5 on an Apple Silicon Mac:

| Historical exported check | Result |
|---|---|
| Shared helpers | 14 passed |
| Inference | 51 tests + 10 subtests passed |
| Training | 84 tests + 36 subtests passed |
| Kernels before final softmax guard | 123 passed |
| Final softmax/calibration-policy regressions | 4 passed |
| Adapter | 26 passed |

A loaded concurrent inference cleanup check failed; isolated supervision and a
full inference rerun passed. One large kernel calibration-policy fixture timed
out after 55 post-guard checks. A reduced real numerical fixture with simulated
idle-host metadata and adversarial softmax checks subsequently passed; this was
policy/correctness evidence, not standard timing certification. Historical CPU
framework traces covered all three native environments and held-out tasks;
alpha1's fresh export rollout covered kernels. See the preserved historical
section of release-checks.json and the immutable alpha1 tag for those records.
