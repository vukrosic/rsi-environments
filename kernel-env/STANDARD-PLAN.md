# First eligible standard kernel episode

This is a workload/reproduction plan, not a measurement or training receipt.
Use it when an inspected accelerator becomes available. Do not stop unrelated
work or repeatedly retry calibration while a known render/game/legacy benchmark
is active. CPU smoke compatibility stays a separate evidence state.

1. Use one frozen installation and preserve its version, Python/torch/NumPy,
   platform, task/scorer hashes and common accelerator-lease path. Keep the
   coding policy, source actions and policy sampling configuration separate
   from the kernel candidate snapshots.
2. Start with one training operation: the shared adapter currently chooses
   RMSNorm and default train seed 14. Get its canonical workload without GPU
   work: `difficulty.choose("rmsnorm", "standard", 14)`. The current v1 sampler
   selects `wide/ordinary`. Do not pick a workload by observing held-out scores.
3. On an inspected idle host, calibrate **only that required family**:

   ```sh
   kernel-env calibrate rmsnorm --device mps --difficulty-id wide/ordinary \
     --output-dir /absolute/path/to/new-calibrations
   ```

   CUDA is an unverified implementation path; start with CPU reproduction or
   Apple MPS where supported. A calibration with blocked status is not usable.
   Accepted status requires three baseline samples per declared dev/final
   shape variant with spread at most 10%, matching source/hardware conditions.
4. A fixed seed/family can use that ordinary receipt directly. For varied
   seeds/families, assemble an index **from existing successful receipts**:

   ```sh
   kernel-env index-calibrations rmsnorm \
     --receipt wide/ordinary=/absolute/path/to/accepted-calibration.json \
     --output /absolute/path/to/new-calibration-index.json
   ```

   Partial indices are intentional. Uncalibrated selections block with null
   reward; do not fill missing families with borrowed times, placeholders or
   learned estimates. Add a family only when that workload is needed and can
   be reliably measured. The index does no GPU work and cannot certify timing.
5. Feed the index or single receipt path through the shared adapter's existing
   `calibration` string field in a kernel-only run. No second adapter is needed.
   For direct native use:

   ```python
   env = KernelEnvironment("rmsnorm", "fresh-work", device="mps",
                           profile="standard", seed=14,
                           calibration="/absolute/path/to/calibration-index.json")
   observation = env.reset()
   # External coding policy supplies source. Its terminal result must be
   # trainable=true, exclude_from_rl=false and genuinely measured.
   ```

6. Compare a working baseline starter control and a candidate with five final
   trials, fresh inputs and independent correctness checks. The scorer must
   accept baseline drift/brackets, captured calibration bytes, source identity,
   shape/dtype/finite-output/input-mutation checks and task semantics. A gap
   within the measured noise is not an optimization finding. Preserve blocked
   and incorrect outcomes instead of selecting only attractive timings.
7. Only after the shared trainer is independently cleared and an actual local
   coding-policy checkpoint is supplied should its owner propose a bounded
   weight-update run. Environment compatibility, valid standard speed reward,
   optimizer weight changes and untouched held-out learning gains are distinct
   receipts. A kernel submission or prompt comparison does not update policy
   weights. Eval-only attention/Newton–Schulz never feed updates or selection.

## Second-machine CPU receipt

After installing this package's dependencies on another machine, run the local
source helper with fresh output paths:

```sh
python examples/reproduction_check.py --output /absolute/path/to/new-receipt.json \
  --workspace /absolute/path/to/new-workspace
```

It records the imported code's actual version/location/hashes and the runtime,
checks a fixed correct softmax smoke episode and a wide zero-mass rejection,
and preserves full native records. It downloads nothing, performs no GPU work
and does not contact anyone. Compare those exact source/dependency identities
with the intended package before describing the receipt as an independent
reproduction. The same-Mac demonstration is not second-machine or adoption
evidence, and a compatible smoke score cannot become a standard RL reward.
