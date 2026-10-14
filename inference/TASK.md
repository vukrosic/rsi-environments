# Task contract v2

Optimize inference for the exact [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
checkpoint in `model.json`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
`assets-manifest.json` records every downloaded asset's SHA-256 and byte count.
The model is loaded locally in float32 with SDPA. Remote model code is not enabled.

Edit one self-contained Python source file defining `Engine(model)`,
`prefill(input_ids)` and `decode(token)`. Prefill resets the request cache and
returns `[batch, vocab_size]` last-position logits. Decode advances the existing cache
by one token and returns the next logits. Inputs are unpadded integer token tensors
`[batch, length]` on the chosen device; decode inputs are `[batch, 1]`.
Timed requests are batch one; small correctness
probes additionally switch between batches one and two without padding. Return stable tensors; the scorer retains
their outputs. Do not mutate input tensors.

Any algorithm, precision, weight layout, cache implementation, or custom kernel
is permitted if it preserves this model's required outputs and meets the resource
budget. Weight packing or conversion during construction is allowed and recorded.
Candidates must operate only on inputs supplied through the Engine interface;
reading scorer inputs, baseline outputs, or timing artifacts, modifying clocks,
or moving scored requests into untimed setup violates the measurement contract.
These are validity requirements, not a restriction to a list of approved methods.

## Cases and correctness

Every call draws a fresh 128-bit nonce. It creates fresh natural-text prompts with
varying words/numbers and uniform random-token stress prompts at lengths 32, 128,
and 512. Model weights are always pretrained; random-token inputs are robustness
cases, not random-weight model evidence. Development and final themes differ and
their seeds are independently drawn. Final cases are unseen in earlier feedback,
but local files are public to the same user; they are not secret holdouts.

Development: 3 trials × 3 lengths × 2 cases, eight autoregressive decode steps.
Final: 5 trials × 3 lengths × 2 cases, eight decode steps. Check: one trial per
length and two cases, four steps; it is correctness-only. Each case also checks
two arbitrary teacher-forced continuation tokens. Distinct random warmup inputs
exercise every length twice without warming scored answers.

V2 adds thirteen untimed cache probes: A → B → A request resets; short and longer
prompts with a repeated prefix; two branches reset to their common prefix; batch
one → two → one; and decode after a batch-two prompt. Inputs are fresh and recorded.
The baseline verifies each against independent uncached full-context logits;
candidate probes must match that verified baseline, including exact argmax tokens.
This expands correctness, not the timed workload. Existing `tasks.json` training
and held-out definitions remain unchanged; V1 receipts retain their original contract.

Check the full vocabulary logits after prefill and every decode step, including
every teacher-forced step. All values must be finite and satisfy
`abs(candidate-reference) <= 0.002 + 0.002*abs(reference)` elementwise.
All argmax token identities and the complete generated token trace must match
exactly. Shapes and case identities must match. A single failure invalidates
the whole submission. The baseline additionally compares the end of each greedy
and forced continuation against uncached full-context model evaluation.
V2 artifacts must cover every saved input case, all 49,152 vocabulary logits and
all thirteen cache probes, with the specified step counts and exact input contexts.
Generated tokens must follow the recorded greedy logits. Agreement between two
incomplete artifacts does not establish correctness. An incomplete reference is
an evaluator error/null; an incomplete candidate with complete references is invalid/zero.

## Reward and budgets

For each length, take medians for prompt-processing time and eight-step generation
time. Baseline time is the mean of its before/after medians. Aggregate the six
baseline/candidate ratios with an equal-weight geometric mean. The standardized
reward is `speedup/(1+speedup)`, so unchanged code is about 0.5 on a stable machine.
An invalid candidate receives 0; infrastructure errors and unreliable timing
receive null in the normalized record. Raw diagnostic ratios are never rewards.
Worker startup, provider/device, unknown-attribution and supervision failures are
infrastructure/null. A source/PID-bound failure receipt or explicit Engine contract
violation is required before an execution failure receives candidate zero credit.
`effect_above_noise` separately distinguishes a speed ratio from a supported gain.

An episode permits ten development/check calls and one final call. Final seals
the episode, including a failed or blocked final; start another explicitly named
episode to repeat an experiment under new conditions. Blocked preflight attempts
consume calls after acquiring the episode lease; a busy lease does not consume one.
Caller-created episode folders are bookkeeping, not protection against a hostile
agent resetting its budget. The training harness must own the episode path.

Each worker: 180-second hard wall deadline, sampled 6 GB process-group RSS ceiling,
two PyTorch CPU threads, supported BLAS pools limited to two threads. MPS driver
allocation is checked at synchronization boundaries against 4 GB. Whole scoring:
600 seconds for the executing controller, covering snapshot creation, asset validation,
input construction, model loading and output grading. A parent supervisor enforces
that deadline and cleans up owned descendant groups; timeout is an excluded
infrastructure error with null reward. Memory checks are sampled,
not OS-enforced peak limits; see the trust boundary. The workload never exceeds
522 tokens per request. Only CPU and MPS are supported in this release.
