# Measurement and trust boundary

The scorer owns inputs, correctness checks, timing wrappers, reward computation,
and episode state. The candidate edits only its Engine file. A frozen reference
worker runs before the candidate worker and again afterward, in separate processes.
All workers use the same inputs and pinned pretrained assets. This prevents an
ordinary model edit from changing the correctness reference in memory.

Public scoring now launches an evaluator controller in an owned subprocess. A
parent guard watches one monotonic deadline for all executing stages, including
source copying, asset hashing, tokenizer/input preparation, model loads and output
grading. It samples process-tree memory and tracks descendants even when workers
create their own process groups. A hang/failure is an excluded error/null result,
and logs/cleanup metadata survive. Outer API directory/result I/O and interpreter
import before the API call are outside that executing-process guard; a framework
tool watchdog remains necessary for host/filesystem failures.

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

Contract V2 also checks untimed prefix/reset/branch/batch-size probes. The baseline
computes each full-context oracle without a KV cache and checks its cached outputs;
the separate candidate process returns the same probe traces for grader comparison.
These checks exercise batch two only at small prompt lengths; timed throughput
remains batch one. Historic V1 replay stays valid for its original recorded checks.
The grader also checks each V2 artifact against saved inputs: every case, complete
vocabulary, greedy/forced step count, token/logit consistency and all thirteen probe
identities and contexts. Two equally truncated outputs cannot establish correctness.
Malformed baseline artifacts exclude evaluation with error/null; malformed candidate
artifacts receive invalid/zero. Saved V2 replay applies the same coverage checks.

Before assigning candidate credit, the before/after oracle outputs must agree.
Oracle disagreement is an evaluator error with null reward, even if the candidate
also differs; the coding-agent episode terminates and is excluded from updates.
The initial candidate comparison is retained as a diagnostic, not credited when
its reference is inconsistent. Worker reference exceptions receive the same
infrastructure exclusion. Worker failures carry bounded stage/source/PID-bound JSON
receipts. Direct submitted exceptions and explicit evaluator-owned Engine contract
violations can receive invalid/zero. Dependency, asset/model load, serialization,
device/provider and ambiguous failures receive error/null. Missing, malformed or
stale receipts, signal exits and supervision time/RSS failures are excluded rather
than assuming the submitted code caused them. A stage alone does not prove causation;
native RuntimeError at an Engine call is ambiguous unless direct raising is observed.

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

Asset manifests require the pinned identity/dtype/attention, required pretrained
weights/config/tokenizer entries, positive byte counts, safe filenames and valid
SHA256 strings; these are explicit checks that remain active with python -O.
The cache is then checked against both byte counts and hashes. This validates the
cooperatively pinned manifest; same-user adversarial replacement remains outside
the security guarantee.

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
thread checks are best-effort under this trust model. Failure receipt hashes/PIDs
prevent ordinary stale-file mistakes; they do not authenticate hostile worker code.

Before training untrusted agents, add separate-user/container execution, protect
grader/oracle files, stream fresh inputs after candidate initialization, and use
trusted external timing with an explicit IPC-overhead contract. Review actual
submissions for measurement-contract compliance. Do not publish these local scores
as adversarially secure or use them for hostile multi-tenant training.

## Integration status

`InferenceEnvironment.step(path)` and `.finalize(path)` return the suite v1 schema.
The evaluation task ID is `rsi.inference.smollm2-135m.v2`; the frozen train/heldout
theme dataset IDs remain V1. `CodingAgentEnvironment` provides reset/submit transitions
and embeds TASK.md, this document and RL-CONTRACT.md with their SHA256 digests.
The standalone wheel ships the same rules beside its private runtime modules.
Coordinator-owned Verifiers 0.4.0 boundary episodes have exercised syntax-invalid
train/heldout actions and exclusion routing. This establishes that narrow framework
compatibility, not coding-policy weight training or learning. Real pretrained CPU
checks passed correctness with timing excluded; MPS performance remains unverified.
Protected evaluation and cross-host reproducibility remain separate deliverables.
