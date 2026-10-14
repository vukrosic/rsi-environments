# Changelog

## 0.2.3 — October 14, 2026, worker failure attribution

- Exclude worker dependency/asset/model/provider/device/serialization failures with
  null rewards. Only source/PID-bound direct submitted exceptions or explicit
  Engine contract violations receive candidate zero credit.
- Validate bounded atomic failure receipts; missing, malformed, stale, signal and
  supervision failures default to infrastructure exclusion. Native-call and wrapped
  provider failures remain excluded rather than inferring a cause from error text.
- Preserve successful output/timing artifacts without failure-receipt I/O.
- Check the two-thread CPU budget after every prefill/decode return, catching
  lazy thread changes that a constructor-only check misses.


## 0.2.2 — October 14, 2026, contract distribution and complete grading

- Ship TASK.md, DESIGN.md and RL-CONTRACT.md in wheels and source distributions;
  embed their content and SHA256 in reset observations so installed policies can
  resolve the complete measurement and isolation rules (review finding INF-10).
- Bind V2 output coverage to saved inputs, including all cases, the full vocabulary,
  continuation counts, token/logit consistency and all thirteen cache probe contexts.
  Incomplete baseline outputs are excluded error/null; incomplete candidates are
  invalid/zero. Saved V2 replay checks the same contract; V1 keeps its original checks.
- Correct stale integration documentation. Framework boundary compatibility,
  coding-policy weight updates and held-out learning gains remain separate claims.

## 0.2.1 — October 14, 2026, follow-up safeguards

- Validate asset manifest identity, required weights/config/tokenizer entries, byte
  counts, safe filenames and SHA-256 format with explicit exceptions. Checks remain
  active under optimized Python; the pinned current cache still validates.
- Serialize reset with source actions, preserving the active episode and transition
  when another thread requests reset.
- Preserve the verified 0.2alpha snapshot and its wheel receipts separately.
  Neither safeguard changes the held-out split, task correctness contract or rewards.


## 0.2.0 — October 14, 2026, local release candidate

- Supervise the whole executing evaluator with one deadline, including preflight,
  snapshots, asset checks, model loading and artifact grading. Retain process-tree
  memory/cleanup records and exclude controller timeouts from policy updates.
- Treat oracle disagreement as evaluator error/null, even when candidate output
  also differs. Preserve diagnostic comparisons and stop the coding-agent episode.
- Add contract V2 cache correctness probes for resets, varying prompt lengths,
  repeated prefixes, branching continuations and batch-one/two transitions. Compare
  against verified independent full-context logits. Timed workloads remain batch one.
- Add hash-verified saved-output replay with no model execution or new reward.
- Add an independent namespaced Python wheel and CLI, with pinned dependencies and
  separately fetched model assets. Exclude weights, episodes, logs and private caches.

The training/held-out theme definitions and pretrained checkpoint remain unchanged.
Sources and test/run receipts are recorded per checkpoint. Package compatibility,
real coding-policy weight updates and measured learning gains remain distinct.
See [release validation](../VALIDATION.md) and [release validation](../VALIDATION.md).

## 0.1.0 — October 14, 2026, initial local implementation

Pinned pretrained SmolLM2-135M, cached SDPA baseline, editable Engine, full-logit
and exact-token checks, separate worker executions, fresh inputs, bounded worker
budgets, suite score records and accelerator lease, coding-agent reset/actions/
termination, train/held-out routing, reproducibility docs, and retained CPU evidence.
The fp16 attempt failed correctness. Busy-machine diagnostic timing did not support
a speed gain. No hosted release, MPS verification or policy weight training occurred.
