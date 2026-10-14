# Validation and evidence scope

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
