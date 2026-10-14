# Training environment changelog

## Source release 1.2 — October 14, 2026

Candidate protocol remains `training-v1.1-fixed-steps`; policy API remains
`training-episode-v1`; dataset semantics remain `training-semantic-v1`.

- Added evaluator-owned process-memory LRU caching of successful semantic
  reports, bound to raw manifest/tokenizer/array hashes and validator source.
  File hashing and basic validation still run for every load. There is no disk
  cache or candidate cache assertion. Local overhead: uncached 9.687/5.317 s,
  cold 6.294 s, warm 0.117 s; filesystem/load conditions were uncontrolled, so
  this is usability evidence rather than a benchmark reward.
- Cache tests cover rehashed array/tokenizer/manifest mutation, canonical dtype
  overlap, decoded overlap, validation-time mutation, caller edits, LRU bounds
  and forged disk assertions. The cache module passed 36 tests, including inherited
  semantic cases. A missing-file error-contract regression found in the linked
  21-case contract run was fixed and rechecked with four related cache cases.
- Preparation refuses occupied destinations; three source-preservation tests pass.
  Persisted episode budgets reject corrupt counts/types/config; three focused
  persistence tests pass, including a terminal interruption that remains sealed.
- Added actual shared Verifiers 0.4.0 checks on frozen native sources. Training
  and held-out smoke episodes complete; null infrastructure truncation is
  excluded. Framework scoring intentionally raises an exclusion error for these
  ineligible samples; `framework_ok=false` is recorded, not hidden. No outer
  policy optimizer ran and no learning gain was measured.
- Added safe source-package inspection and isolated-venv reproduction. Two
  package-safety tests pass. A fresh venv on this Mac installed pinned torch
  2.8.0, numpy 2.2.5 and tokenizers 0.22.1 without system site packages; its
  complete source-package/data-preparation check is recorded in the continuous
  checkpoint. This is not another-machine evidence.

Current receipts: `checkpoints/2026-10-14-validation-cache/measurement.json` and
`checkpoints/2026-10-14-continuous/`. Historical data is unchanged. Shared suite
implementation is read-only here; compatibility runners use the coordinator's
installed adapter and copies of native/shared source. No duplicate trainer adapter
was built. Standard GPU calibration/training, outer coding-agent weight updates,
matched learning improvement, hostile-code isolation and fresh-machine portability
remain unverified.

## Source release 1.1 — October 14, 2026

Closed rehashed same-token-stream/different-integer-dtype held-out leakage;
validated tokenizer/story semantics and whole-story identities. Added phase-aware
child failure classification, post-validation time-cap enforcement, explicit
reward eligibility and restored feedback. 40 CPU tests passed; real-data smoke
and backend JSON-lines checks succeeded. See the immutable semantic-v1 checkpoint.

## Initial source release 1.0 — October 14, 2026

Introduced editable 5,063,424-parameter GPT starter, independent train/grade
workers, dev/final data preparation, normalized suite records, compute limits,
calibration gate and persistent code-action episodes. 21 initial CPU tests passed,
with synthetic and real smoke examples. Historical experiments were preserved.
