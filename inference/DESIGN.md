# Measurement and trust boundary

The scorer owns inputs, correctness checks, timing wrappers, reward computation,
and episode state. The candidate edits only its Engine file. A frozen reference
worker runs before the candidate worker and again afterward, in separate processes.
All workers use the same inputs and pinned pretrained assets. This prevents an
ordinary model edit from changing the correctness reference in memory.

Loading, asset verification, engine construction, and shape warmup are excluded
from steady-state timings but retained in wall/worker artifacts. Prompt timing
includes cache initialization and output projection. Decode timing includes
argmax, KV-cache updates, eight next-token calls, and clones needed to preserve
the checked outputs. Tensor transfer to CPU, input generation, tokenizer execution,
and teacher-forced/uncached checks are outside the timers. Synchronization brackets
every MPS interval. Reported decode tokens/s counts next-token calls, not EOS-aware
user-facing generation; all requests run a fixed number of steps regardless of EOS.

The baseline is already cached and uses PyTorch's SDPA and `logits_to_keep=1`.
The reference's cache is checked against full-context inference. The official
[Transformers caching documentation](https://huggingface.co/docs/transformers/en/cache_explanation)
and installed `LlamaForCausalLM.forward` were inspected for this implementation.

Before assigning candidate credit, the before/after oracle outputs must agree.
Oracle disagreement is an evaluator error with null reward, even if the candidate
also differs; the coding-agent episode terminates and is excluded from updates.
The initial candidate comparison is retained as a diagnostic, not credited when
its reference is inconsistent. Worker reference exceptions receive the same
infrastructure exclusion; candidate exceptions under the budget remain invalid/zero.

The suite's `accelerator_lock` covers every MPS operation through `evaluate`,
including correctness. The standalone fallback copies the suite's MIT helpers
and uses the same default `~/.cache/vuk-research-envs/accelerator.lock`. It is a
cooperative lock. Historical scripts and unrelated apps may not participate.
`worker.py` is an internal child entrypoint, not an independently authorized GPU
timer; invoke the scorer/interface so the lease exists.

Before, between, during (every five seconds), and after worker execution the scorer
records process load. Known games/renders and heavy CPU processes invalidate timing.
Without `--diagnostic`, an initially busy machine blocks before model execution.
No user processes are stopped. Reference bookend median drift must be at most 5%,
and baseline relative median absolute deviation must be at most 5% in every phase.
Noise floor is the maximum of 5%, bookend drift, and twice relative MAD. This is
a conservative local calibration protocol, not proof of an idle GPU or a published
confidence interval. A supported improvement must exceed that floor and reproduce.

## Records

Every scored call stores source/evaluator copies, input tensors, raw output/timing
tensors, worker stdout/stderr, sampled memory, machine conditions, JSON results,
and a normalized suite record. Source/input/evaluator hashes are rechecked after
execution. The controller records hashes captured at import and fails if those
files changed during scoring. Worker execution uses the copied worker/assets/reference
files. Python/package imports remain supplied by the pinned interpreter environment.
Only one candidate source file is supported; adding external project imports is
outside this first release's reproducible submission closure.

`interface.py` returns the normalized record: `ok`, `invalid`, `blocked`, or `error`.
Feedback persists in `feedback.jsonl` (raw) and `records.jsonl` (normalized).
Events such as a busy lease, sealed episode, or missing file also get durable
individual records. `check`/diagnostic correctness on pretrained weights remains
standard-profile evidence with blocked timing; it is not a synthetic smoke result.

## Isolation limits

This scorer executes trusted local Python with the same user permissions. It is
not a security sandbox. A hostile candidate can inspect the filesystem, read
oracle artifacts, replace shared module functions, forge its worker's timers or
outputs, spawn detached processes, or defeat sampled limits. Hash checks detect
ordinary concurrent edits, not an attacker changing files and restoring them.
The parent monitors the scorer-created process group and kills that group on exit;
it does not control a process deliberately escaping it. Resource limits and CPU
thread checks are best-effort under this trust model.

Before training untrusted agents, add separate-user/container execution, protect
grader/oracle files, stream fresh inputs after candidate initialization, and use
trusted external timing with an explicit IPC-overhead contract. Review actual
submissions for measurement-contract compliance. Do not publish these local scores
as adversarially secure or use them for hostile multi-tenant training.

## Later integration

`InferenceEnvironment.step(path)` and `.finalize(path)` return the suite v1 schema.
The task ID is `rsi.inference.smollm2-135m.v1`. An adapter could give an agent its
editable workspace plus these tools, retain episode state outside that workspace,
and use normalized rewards. CPU is checked here; MPS must be explicitly verified.
There is no tested Prime Intellect Verifiers/OpenEnv adapter or training loop and
no CUDA claim. Those are separate deliverables, along with protected evaluation
and cross-host reproducibility; copying an API shape is not an integration test.
