# Kernel timer v5: known reward-integrity limits

Source review on October 16, 2026 found two timing-integrity gaps in the
published `bench3.py`, SHA-256
`5462cde75bf1e6f791b49f373f7a663eccc793d95f95396d8b9ad6e1d739a3e0`.
Keep this version's speedups out of RL updates and incumbent promotion until
the gaps are repaired and the replacement is validated.

- **Upcoming timed inputs are predictable.** The untimed warm-up seed is an
  invertible transform of the next timed seed. Submitted code can derive the
  upcoming inputs and may compute their answers during warm-up. Correct output
  checks alone cannot establish when those answers were computed. See
  [the warm-up and timed calls](bench3.py#L487).
- **The fingerprint list has no required length.** Its comparison uses `zip`,
  so an empty or short list skips some or all comparisons with the values
  reported inside the timed window. Output correctness checks still run, but
  the timed commitment is incomplete. See
  [the fingerprint comparison](bench3.py#L518).

These are source findings. The described bypasses were not implemented or
run during the review; their speedups and device feasibility are unmeasured.
No historical run is claimed to have exploited either gap.

The [v5 check record](VALIDATION-V5.json) remains a record of its tested cases.
Passing those checks does not rule out these additional paths. Historical
examples remain development measurements, without qualified timing rewards
or evidence of LLM learning. This advisory concerns the standalone kernel
benchmark; it does not change the separate Training Improvement reward.
