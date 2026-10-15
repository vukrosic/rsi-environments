# Fresh public data, CUDA training and a measured reward

This one-command example is **source-prepared and awaits remote end-to-end
verification**. It downloads bounded official text or verifies its raw cache,
prepares a new corpus, trains the fixed small GPT for500 calls and independently
reloads/scores its final weights. It requires no private prepared data or model
checkpoint. It reuses the existing native training/grading code; no LLM optimizer
or automatic recipe search is included.

The advertised target is the observed Linux x86_64 / single NVIDIA RTX A2000
lane, Python3.12.14, Torch2.11.0+cu128/CUDA12.8, NumPy2.5.3 and tokenizers0.22.1.
[requirements-cuda-observed.txt](requirements-cuda-observed.txt) records those
minimum exact dependencies. It is an observed runtime, not a verified fresh
package-install recipe. Use an existing compatible environment/explicit overlay;
the command does not install packages, change drivers or download model weights.
Execution evidence covers A2000 only. Other single NVIDIA devices may attempt
the same fixed protocol with these dependencies; that is not a claim of verified
support. The script records the actual UUID, PCI address, name, driver and memory
at admission and requires that full identity to stay unchanged through the run.
Dependency mismatches still fail closed. Multi-GPU scheduling is not included.

## One command on your GPU host

Keep `training-env/` with its complete `recipe_pair_source/` runtime from the
same source export. From `training-env/`:

```bash
python3 public_cuda_example.py --python "$TRAINING_PYTHON" \
  --overlay "$TOKENIZERS_OVERLAY" --raw-cache "$RAW_CACHE" \
  --output "$NEW_RUN" --idle-confirmed
```

Set those variables to your own existing interpreter/optional dependency overlay,
raw cache and a fresh job directory. Omit `--overlay` if unnecessary; repeat it
for multiple existing overlays. Cache/job paths must be disjoint and outside
the component source tree. The output must not exist. The script preserves venv
interpreter symlinks, clears inherited Python paths/home for children and disables
user site packages. All downloads and large outputs remain on this host.
Use optional `--expected-gpu-uuid` to pin your device explicitly. Our validation
job pins the exact previously recorded A2000 UUID; public users need no A2000
model-name match or private operator identifier.

`--idle-confirmed` is your assertion that other workloads are absent. Use the
same canonical common accelerator lock for participating jobs on this host:
`~/.cache/vuk-research-envs/accelerator.lock`. Native calibration/evaluation acquire
it themselves; no outer nested lease is held. The command rejects observed GPU
compute apps and preserves unrelated processes. Empty boundary inventories are
not proof of continuous idleness; the unchanged strict native calibration/load
checks remain mandatory. No unrelated processes are stopped.

For a source/runtime/device check without corpus transfer, preparation, GPU
tensors or training, add `--preflight-only`. It still queries NVIDIA metadata and
probes the cooperative lease, so run it on the intended host. This check is not
a training result or evidence of a fresh installation.

## What runs

1. Check all explicitly pinned source dependencies, the observed dependency lane
   and actual single-NVIDIA metadata. Capture UUID/name/driver/memory/PCI identity
   for before/after binding, and enforce an optional operator UUID pin.
2. Run the [public raw-data v2 preparer](RAW-DATA.md), requiring native input
   hashes to match the original raw lock and post-preparation/final cache checks.
   The official revision, byte selection, tokenizer and publisher license/card
   rules remain those of `raw-data-profile.json`. No full-train fallback or retry.
3. Obtain a fresh strict CUDA calibration, then run the unchanged audited LR0.001
   starter on the new corpus:500 calls, B32/T256, 4,096,000 training tokens,
   seed19001. An independent process scores491,520 next-token predictions from
   raw dev span[0,493440). `final.npy` is integrity-checked, never loss-graded.
4. Run the existing saved-state CPU audit: actual CUDA worker/grader metadata,
   initial/final canonical and file hashes, changed inner-model weights and
   independent grader-loaded final checkpoint agreement. Check shared TF32 policy,
   unchanged source/data/calibration and final physical/common-lease controls.

The new corpus has its own content identity. This command does not reconstruct
the accepted92 bytes or its historical losses. Model shape is fixed at5,063,424
parameters; max_params5,100,000 and max_tensor_bytes64,000,000 are unchanged.
Training uses only train text; validation never fits the tokenizer. The score
is a development loss, not an untouched final-test result or evidence of learning
improvement over a matched baseline.

The whole supervised workflow has1200 seconds of work, deducting small source
setup elapsed. Phase caps are raw preparation300s, calibration90s, native
episode600s and saved audit30s, within the remaining shared budget and15s cleanup
reserve. The existing process supervisor tracks/cleans owned descendants and
records its separate bounded cleanup tail. Filesystem setup is not a hard
realtime bound. Initial free disk/configured thread pools and recorded RSS/CUDA
allocator peaks are not hard RAM/VRAM/disk/thread quotas; admit adequate resources
before execution and preserve actual observations. One training worker and one
independent grader run sequentially, plus the required calibration/reference work.

## Inspect the result

Read `$NEW_RUN/result.json`, `workflow.json`, `whole-job-supervisor.json`, phase
supervisors and physical controls. Exit0 alone is insufficient. Full completion
has `completed_pending_independent_review`, the new workload/manifest identity,
measured `loss` and `native_reward`, plus `episode/record.json` and
`saved-audit.json`. Native reward is exactly `exp(-loss)` after native correctness
and resource controls. The result preserves initial/final tensor hashes and says
that no LLM weights, promotion or reward consumption occurred. An independently
accepted session is a separate evidence state, not set automatically by this tool.

Failures stay blocked with null effective reward and retained partial outputs.
Do not treat missing dependencies, download/integrity/calibration failures,
timeouts or missing cleanup proof as a candidate-quality penalty. Preserve raw
native records, including any invalid/negative outcomes, without promoting them
to a successful reproduction. No automatic retry or relaxed threshold exists.

Raw text, publisher card, arrays, checkpoints and full logs stay on the GPU host.
Consult TinyStories' declared CDLA-Sharing-1.0 terms before redistribution; the
source export includes neither datasets nor private job manifests/paths. Share
compact reviewed controls rather than uploading weights or raw text by default.

The source artifact must explicitly include **both** `raw-data-profile.json` and
`public_cuda_example-spec.json`, in addition to the complete pinned Python/native
runtime. A missing JSON or helper is an incomplete export and blocks execution.
See [INSTALL.md](INSTALL.md) for the separate exact historical-asset route.
