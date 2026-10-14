# Changelog

## 0.1.4 — October 14, 2026 (local development)

- Native calibration indices route the shared trainer's existing path field to
  the seed-selected training family. Partial indices are supported; missing or
  mismatched receipts stay blocked/null. Ordinary receipts remain compatible.
  Exact parsed receipt/index bytes are hashed and preserved, closing the prior
  separate-read/hash race. **95 pure routing fixtures** and **17 native gate
  checks** passed; those are not fabricated hardware measurements.
- Added a portable CPU reproduction checker, fixed source-versus-installed
  version attribution, refusal to overwrite outputs/workspaces, and complete
  artifact/source/runtime receipts. **5 focused fixture tests passed**; the
  initial checker completed an actual same-Mac CPU run. The final exported
  checker also passed against installed 0.1.4 in a fresh isolated environment
  with newly downloaded dependencies and system packages disabled. Second-machine
  reproduction and external adoption remain unverified.
- Added `STANDARD-PLAN.md` for one needed idle family, index assembly from real
  existing receipts and correct distinction between environment compatibility,
  calibrated standard reward, weight updates and held-out improvement. No
  kernel accelerator execution, policy-weight update or public staging change
  was made. A bounded MPS calibration attempt was rejected before execution
  because Codex exceeded the declared unrelated-process CPU-load limit.
- Corrected the standard seed-14 plan to the executed sampler result,
  `wide/ordinary`, and embedded supported verification scope directly in README
  so the exported source does not link to an omitted private status document.

## 0.1.3 — October 14, 2026

- Softmax correctness now verifies probability bounds and row normalization.
  Missing/zero probability mass no longer passes a loose per-element tolerance.
- An explicit malformed submit seals the native episode with an unmeasured
  parser-invalid zero receipt; final actions cannot receive free retries.
- Replaced a large standard no-calibration policy test with a reduced synthetic
  correctness fixture plus simulated idle state. The broad run recorded
  **141 passed / 1 timed-out test** on this loaded machine; that timeout was a
  correctly excluded execution error. The corrected gate plus changed scorer,
  exclusion and budget tests passed **54 checks in 30.76 seconds**. Failure
  source/log/trial evidence is retained locally rather than hidden or relabeled.
- Source release uses an explicit data-free allowlist, deterministic archive
  metadata, SHA-256 manifest and overwrite refusal. Source packaging tests
  passed **2 checks**; final wheel/source membership receipts are retained in
  the dated backlog checkpoint. No release artifact has been published.

## 0.1.2 — October 14, 2026

- Added canonical, reproducible training-only shape/distribution sampling:
  nine families per task, bounded allocations, actual case metadata and local
  RNG. Held-out attention/Newton–Schulz inputs remain unchanged. **38 focused
  sampler checks passed in 3.84 seconds**.
- Added strict native action budgets, no-write-after-deadline behavior,
  parser-invalid feedback, configuration validation, modular development seeds
  and reset replay. **18 adverse lifecycle checks passed in 2.90 seconds**.
- The independently installed 0.1.2 wheel passed **five adverse checks** from
  outside the checkout. A real infinite-loop candidate printed an entry marker
  before being stopped at its eight-second timeout (**8.213 seconds observed**);
  its null reward was excluded. No GPU was used.
- Trusted baseline protocol faults and trusted scorer changes now block with
  null rewards. Malformed/nonfinite/overflowing calibration latencies and wrong
  difficulty receipts are rejected; failed final calibration integrity checks
  cannot leave `status=ok`. Selected-family shapes are consistent in task specs.
- Actual shared Verifiers 0.4.0 train smoke rollout performed two code actions
  using the sampled `odd/offset` RMSNorm family. Native final correctness passed;
  both native and framework layers excluded its smoke reward from RL updates.

## 0.1.1 — October 14, 2026

- Added native/raw-record training eligibility markers, evaluation partition
  provenance, persistent transitions, and structured public task specifications.
- **65 CPU tests passed**. Installed-wheel invalid training submission retained
  numeric zero. Shared Verifiers 0.4.0 held-out compatibility rollout honored the
  native exclusion markers. No policy weights were trained.

## 0.1.0 — October 14, 2026

- Six kernel tasks, editable starters, independent mathematical scorer,
  bounded subprocess runner, fixed idle calibration protocol, normalized suite
  records, source snapshots and budgeted reset/step/submit episodes.
- **28 CPU tests passed**, and a wheel installed outside the checkout completed
  a CPU rollout. MPS standard timing was blocked by other workloads; CUDA was
  not tested. Numerical smoke scores are development checks, not learning or
  reliable optimization results.
