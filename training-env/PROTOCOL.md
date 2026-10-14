# Protocol: training-v1.1-fixed-steps

October 14, 2026. Versioned changes from historical `rsi/score.py`: separate
story-disjoint dev/final arrays, independent weight reload/loss calculation,
validated shapes/finiteness/causality, bounded process runtime, normalized
score records, persistent interaction budget and calibration guard.

| Setting | Standard | Smoke |
|---|---:|---:|
| Vocabulary | 4096 | 4096 |
| Maximum registered parameters, including frozen | 5,100,000 | 5,100,000 |
| Starter parameters | 5,063,424 | 5,063,424 |
| Batch / context | 32 / 256 | 2 / 16 |
| Train-step calls | 500 | 3 |
| Delivered training tokens | 4,096,000 | 96 |
| Held-out batches / tokens | 60 / 491,520 | 2 / 64 |
| Training elapsed-time cap | 300 seconds | 30 seconds |
| Per-child hard watchdog | 600 seconds | 90 seconds |
| Registered parameters + buffers | 64 MB | 64 MB |
| Checkpoint cap | 100 MB | 100 MB |

The fixed work budget is delivered batches and train-step calls, **not equal
FLOPs**: candidates can vary architectures, schedules, precision, and optimizer
work. The wall cap is an additional device-dependent resource limit. Compare
only the same profile, seeds, data manifests, device and calibration identity.
Do not compare this version's rewards directly with historical batches. Smoke
uses the same useful model scale but tiny batches and contexts. Real-data smoke
is still smoke, not a standard benchmark.

## Candidate interface

Edit a copy of `starter.py`:

```python
build_model()                  # torch.nn.Module, returns logits[B,T,4096]
build_optimizer(model)         # candidate-controlled optimizer object
train_step(model,opt,x,y,frac)  # any update method; return value ignored
```

No optimization idea is banned beyond correctness/resource constraints. The
model must be causal: changing future tokens or trimming a sequence must not
change prefix logits. Deterministic eval mode is required. Registered weights
and buffers and all evaluated logits must be finite. At least one registered
parameter must change during training. The evaluator owns batches, labels,
500 calls, seeds and loss calculation. `frac=step/steps` preserves the historical
schedule, including the zero-learning-rate first step in the starter.

The training worker gets only the training path. A fresh grading process
constructs the candidate model, reloads `weights_only=True` checkpoint state,
checks validity, and recomputes float32 cross-entropy over the selected held-out
split. It never uses the train-step return, stdout claims, or a candidate metric.
Frozen source and data hashes are checked after execution. All runs write a
normalized `suite.records.make_record` envelope. Invalid correctness/resource
results have reward 0; infrastructure blocks/execution errors have null reward.

## Data and interaction

Prepare with `prepare.py`: train-only BPE, complete EOT-delimited stories,
validation divided near its token midpoint into physically separate dev/final
files. Cross-split duplicate story hashes fail preparation. Manifest hashes,
token ranges, ranks, counts and tokenizer vocabulary are checked before a run.
Standard requires real assets; there is no random-data fallback. Synthetic
smoke data requires an explicit preparation switch and is labeled in each record.

Dev and final start at offset zero in their **different** streams. Seeds govern
initialization and training samples, not which data split is evaluated. Every
submission retrains from scratch; final retrains the frozen last submitted code
and scores the terminal split: dev for policy-training episodes, final for held-out
policy-evaluation episodes (see RL-INTEGRATION.md). No best-of-dev checkpoint is carried over. Final is
terminal. Invalid candidate submissions also consume a dev attempt; infrastructure failures
truncate the episode and have no reward; reset is an observation,
not a budget reset. Use multiple predeclared seeds and unchanged baselines for
research; this deliverable's small examples do not establish an improvement.

## Timing and hardware

All accelerator work (training, grading, reference calibration) holds the shared
`suite.resources.accelerator_lock` lease. CPU checks do not claim GPU verification.
Known heavy applications block standard runs without being stopped. Calibrate on
an idle device, then each standard run remeasures the unchanged starter and
requires its median within ±10% of calibration and relative spread ≤10%.
Calibration expires after 24 hours. The detector is incomplete; the operator's
idle assertion is evidence, not proof. Record other workloads for scientific use.

```bash
python calibrate.py --device mps --idle-confirmed --output runs/idle-mps.json
python interaction.py step --session runs/standard --data data --candidate starter.py \
  --profile standard --device mps --calibration runs/idle-mps.json
python interaction.py final --session runs/standard --data data --candidate starter.py \
  --profile standard --device mps --calibration runs/idle-mps.json
```

## Trust boundary and release

Subprocesses and cooperative locks are **not security isolation**. Candidate
Python can access host files, import modules, patch Python/Torch state, write
files, or allocate unregistered memory. Causal checks sample behavior and are
not a proof of causality. Public final assets are held out from the interface,
not secret. The persistent budget can be bypassed by a host user creating a new
session; a trainer must bind session allocation to the experiment's agent ID.
For adversarial use, run candidates in a disposable container/VM without held-out
mounts and a separately trusted architecture/loss service, enforce CPU/memory/
network limits externally, and prevent scorer/data mutation. Never give untrusted
agents direct access to this evaluator on a personal machine.

Keep data, raw source text, checkpoints and private agent logs out of a release.
Preserve evaluated source snapshots and normalized receipts; checkpoints are
local ignored reproduction aids. Standalone redistribution must include
`suite/{__init__.py,records.py,resources.py}` from the same revision (MIT notice
in each helper), alongside this folder. Or copy the whole suite and training
folders into a minimal `rsi/` tree. The history/batches/dataset are not dependencies.
Supported host platforms are Linux and macOS (`fcntl`/process-group watchdog).

## Revision v1.1 — October 14, 2026

Semantic validation `training-semantic-v1` now runs over every real split at
scoring, including real-data smoke. It validates deterministic preparer-compatible
ByteLevel BPE, the special EOT and byte alphabet, whole nonempty terminated
stories and declared story counts. It rejects cross-split canonical uint16 token
story identity and whitespace-normalized decoded-text identity. Dtype/file-header
changes cannot hide the same story. It does not authenticate dataset origins or
train-only tokenizer fitting; partial-story overlaps and paraphrases remain outside
this exact-story check. Synthetic smoke explicitly skips story semantics.

The measured 30.38M-token local data check took 10.744 seconds on CPU. Validation
uses 65,536-token mmap chunks and compact digests; raw retained digest size is
64 bytes per story (9,576,512 bytes at this scale), with additional sorting/unique
arrays and tokenizer/runtime memory. Decoding a story is capped at 1M tokens.
Each evaluation performs validation before and after execution, so roughly twice
the single-check cost occurs; these checks are outside the candidate loop cap.
Their elapsed times are recorded in provenance. No cache or origin attestation is
claimed. Files/manifests are rehashed after execution; timing metadata can differ
without being mistaken for a dataset mutation.

The unchanged 300/30-second candidate caps now include per-step model validation
and final causal/logit validation, closing the unchecked last interval.
`train_seconds` records loop elapsed time; `train_with_final_validation_seconds`
includes final checks. Each child still has its own 600/90-second total watchdog;
an episode can include training plus grading watchdogs, calibration and two data
checks, so the per-child watchdog is not an episode-total deadline. Calibration
identity changes with this protocol version; use a fresh idle calibration.

Child results label execution phase. Missing/unreadable assets and storage errors
are `error`/null reward; recognized lost/unavailable device runtime is
`blocked`/null. Syntax/shape/config failures and generic candidate runtime errors
are `invalid`/zero. Device-side assertions count as candidate failures; candidate
OOM counts as a resource failure. Runtime exception attribution is necessarily
cooperative and incomplete: a generic host failure during a candidate hook may
be ambiguous. The outer trainer must honor infrastructure exclusions.

## Hash-bound validation cache — October 14, 2026

Successful real semantic reports can be reused only in the evaluator process
memory (LRU of four entries). Every load still rehashes manifest, arrays, tokenizer
and validator source, enforces basic token/rank/count checks, and checks byte
snapshots again before returning. The complete file snapshot, source hash and
semantic constants bind the key. Recomputed malicious manifest hashes therefore
trigger full analysis; failed validation is never cached. No disk cache or
candidate-supplied assertion is accepted. Use `load_data(..., use_cache=False)`
for a full audit. Reports are copied so caller edits do not alter cached results.

The bounded local check measured uncached 9.687/5.317 seconds, then cold 6.294
seconds and warm 0.117 seconds. CPU/filesystem caches and other work were not
controlled; this demonstrates practical loader overhead, not a scientific speed
reward. Before/after receipts are in the validation-cache checkpoint. The earlier
no-cache statement describes the prior semantic-v1 checkpoint. Causality and
held-out-loss scoring still run for every candidate; this caches data semantics
only. The candidate compute protocol remains v1.1.
