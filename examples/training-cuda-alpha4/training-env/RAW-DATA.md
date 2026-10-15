# Fresh data from the publisher's text

This path prepares a **new workload** from official TinyStories text without
private token arrays. Source is prepared for review; remote execution and fresh
data preparation have not yet been verified. It reuses `prepare.py` and `data.py`
unchanged. It does not start training or update an LLM.

## Accepted92 is a separate identity

The accepted92 manifest records conversion of existing token streams and their
hashes. It does not record an original raw-text revision, text selection or
tokenizer-build provenance. The documented inputs therefore cannot establish
byte-for-byte reconstruction of accepted92. Its exact arrays and tokenizer are
still required by the frozen portable CUDA recipe. That recipe continues to
reject these fresh outputs; do not change its asset hashes to make them fit.
No public distribution route for the historical five files is supplied by this
source handoff. The historical wrapper requires separately supplied authorized
assets; fresh public reconstruction of its result remains unsupported.

The new family is `tinystories-v2-gpt4-prefix100mb-data-v1`. A successful preparation
adds a content-derived suffix binding the measured raw selection, source profile,
dependency versions, tokenizer and three token-array hashes. This is a distinct
future workload; no experiment or learning result is implied.

## Pinned source and selection

[raw-data-profile.json](raw-data-profile.json) is the complete machine-readable
specification. It pins publisher `roneneldan/TinyStories`, immutable revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, the GPT4 V2 train and validation
filenames, their LFS full-file SHA-256/byte counts, and the selection below.
The dataset card identifies [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
The wrapper preserves the publisher card, source attribution, license reference
and a modification notice; consult the terms before redistribution.

- Train: exactly bytes 0–99,999,999 of the 2,227,753,162-byte file. HTTP206 and the
  exact Content-Range are required before reading the body. A server ignoring
  Range is blocked; there is no full-train download fallback.
- Validation: the entire 22,502,601-byte file, checked against its publisher LFS
  SHA-256. Both small LFS pointers must match the pinned revision before data
  transfer. The pinned card's license metadata must match before large transfer.
- The selected train prefix has no independently supplied publisher prefix hash.
  First acquisition binds the pinned HTTPS origin/revision, matching full-file
  LFS pointer, range headers, size and a newly measured prefix hash. This does
  **not** verify the full train blob's contents. `raw-lock.json` explicitly says
  so. Cache reuse checks the measured hashes of every selected file again.

The existing preparer decodes UTF-8 with invalid bytes ignored, keeps complete
EOT-terminated stories, strips their surrounding whitespace and discards a
partial trailing story. It trains a 4096-token ByteLevel BPE on every fourth
training story. Validation splits at the whole-story boundary nearest the token
midpoint. Existing normalized-text and token-story duplicate exclusions, array
integrity and standard minimum sizes remain mandatory. Failed integrity or split
checks block preparation; no filtering, threshold relaxation or candidate penalty
is substituted.

The wrapper compares the unchanged native preparer's recorded `train_text` and
`validation_text` hashes and paths with the original raw lock. It reverifies all
cached texts, the card, both pointers and the raw lock after preparation, then
again after semantic validation/card copying before saving a successful workload.
A self-consistent replacement cache/lock cannot replace the original identity.
Any disagreement blocks with null reward and preserves partial output. These
before/after checks assume a cooperative filesystem; a hostile writer that changes
and restores bytes during reads is outside the guarantee. The v2 correction and
its new fixtures await independent source review and remote execution.

The profile records the actual tokenizer configuration, including its unsorted
library-provided initial alphabet and absence of an explicit RNG seed. Identical
tokenizer bytes across retraining are **unverified**. Preserve the actual
serialized tokenizer hash and all prepared assets for subsequent runs. Two
preparations producing different bytes receive different workload identities.

## Run on the intended compute host

Use an existing Linux CPU interpreter with Python 3.12.14, NumPy 2.5.3 and
tokenizers 0.22.1. These are the observed remote dependency versions, not a verified
fresh installation. There is no Torch import or GPU computation in this path.
No packages are installed and provider Torch/drivers are untouched.

From `training-env/`, with paths supplied by your operator:

```bash
python3 prepare_public_data.py --python "$PREP_PYTHON" \
  --overlay "$TOKENIZERS_OVERLAY" --cache "$RAW_CACHE" --output "$NEW_PREP_JOB"
```

Omit `--overlay` when unnecessary; repeat it for explicit dependency overlays.
The launcher clears inherited Python paths and disables user site packages for
the child. Choose **disjoint cache/job directories outside the source tree**;
the job directory must not exist. The child uses at most two configured worker
threads, requires at least 1 GiB free disk initially and defaults to a 300-second
child work watchdog through the existing process supervisor. `--work-seconds`
may reduce that child budget, never increase it above 300 seconds. It does not
enforce the entire parent launch/staging/cleanup envelope; an operator must bind
that external watchdog separately before dispatch. Acquisition has a 90-second
chunk deadline and at most 10-second socket waits. Cleanup has its existing
separate bounded tail. This is cooperative supervision, not a hard RAM cap or
hostile-code sandbox. Raw text plus prepared arrays stay on the compute host.

Optional modes:

```bash
# Acquire and bind raw text only; uses stdlib, skips tokenizer/dependency imports.
python3 prepare_public_data.py --python "$PREP_PYTHON" \
  --cache "$RAW_CACHE" --output "$NEW_RAW_JOB" --acquire-only

# Reuse a complete hash-checked raw cache with no network fallback.
python3 prepare_public_data.py --python "$PREP_PYTHON" \
  --overlay "$TOKENIZERS_OVERLAY" --cache "$RAW_CACHE" \
  --output "$NEW_OFFLINE_JOB" --offline
```

For a raw-cache-only check, combine `--offline --acquire-only --work-seconds 30`.
The 30 seconds covers child work; setup and the supervisor's bounded cleanup tail
remain outside that work timeout. Fixtures and an aggregate workflow limit also
require the executor's own bound envelope. JSON request values alone do not
establish those external controls.

The cache subdirectory is keyed only by the pinned raw-source/selection
definition, enabling reuse across wrapper revisions. Existing incomplete caches
and `.part` files are preserved and block reuse. Select a new cache root to retry;
the wrapper does not delete, overwrite or retry partial acquisitions. A completed
cache's lock is local integrity evidence, not publisher authentication against a
malicious cache editor. HTTPS redirects must remain HTTPS; signed redirect URLs
are never recorded.

## Retain the evidence

`result.json` records the child/supervisor outcome; failures remain infrastructure
blocks with null reward and `agent_trainable:false`. `phases.jsonl` identifies the
last reached phase; `child-error.json` records the failure class. A successful
job additionally contains:

- `raw-lock.json`: selected-file hashes, pinned upstream specification and bounded
  HTTP metadata. The same lock lives beside the cached raw files.
- `data/`: native five-file corpus, unchanged native manifest, publisher card and
  modification notice. The native manifest retains private input paths. Do not
  publish it without review or claim a sanitized copy has the original hash.
- `workload.json`: portable content identity excludes machine paths and time;
  the separate native manifest hash binds this particular preparation receipt.
  The v2 raw-input proof binds both preparer input hashes and the post-use cache
  to the original raw identity.
- `child-result.json`: full native standard semantic validation, actual tokenizer
  and array identity through `workload.json`, and a separate capacity flag for
  two 493,440-token dev windows. A false flag does not relax a later paired task's
  requirement. `final.npy` is integrity-checked, never loss-graded by preparation.

Actual interpreter/prefix and NumPy/tokenizers import origins are recorded in
private runtime controls, separate from the portable identity. Version agreement
is required but does not replace review of those resolved import paths. Initial
free disk and configured thread counts are guards, not hard RAM/disk/thread caps;
the executor must check current host headroom and retain resource observations.

No training, reward consumption, promotion, accepted92 reproduction or held-out
learning gain is claimed by any preparation result. Keep failed and partial
outputs. A future run must freeze this new workload and its own source/runtime/
resource controls; no new experiment is authorized by these commands.
