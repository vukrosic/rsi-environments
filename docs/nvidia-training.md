# NVIDIA training: reproduction steps and prepared outer-policy example

The native training environment has a CUDA code path. **NVIDIA calibration,
standard execution and outer coding-policy training have not been verified in
this release.** The external Prime-RL example passed Mac configuration/schema,
Verifiers API loading and synthetic training-admission checks. Those checks do
not establish Linux ML runtime compatibility, tokenized rollouts, policy weight
updates or learning gains.

The separate [Apple GPU check](apple-gpu-check.md) records one real-data MPS
inner-model smoke. That supports functional Apple GPU execution at audited tensor
boundaries, with smoke rewards excluded from policy training. It does not certify
standard timing or NVIDIA execution. Inference optimization supports CPU/MPS;
**inference-environment CUDA is unsupported**. Training and selected kernel tasks
have CUDA source paths that still need actual NVIDIA verification.

## Native TinyStories training on one NVIDIA GPU

This runs the environment's inner GPT, not an outer coding-policy optimizer.
Follow [training-env/INSTALL.md](../training-env/INSTALL.md) for the isolated
Python environment and real TinyStories preparation. Keep `training-env/` beside
`suite/`. Use a CUDA-enabled PyTorch build compatible with the actual Linux driver;
the tested Mac dependency stack alone cannot demonstrate that compatibility.
Keep the prepared `train`, `dev`, `final` and tokenizer files plus their manifest.
Synthetic data is rejected by the standard profile.

On the intended idle NVIDIA machine, choose a real device from its inventory.
The commands below select physical device 0; replace that selection if needed.
From `training-env/`, with the native Python environment activated:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# Only assert idle after checking that competing heavy jobs are absent.
# Choose new paths: preserve existing calibration and session receipts.
python calibrate.py --device cuda --output runs/cuda-calibration.json --idle-confirmed
python interaction.py reset --session runs/cuda-standard --data data \
  --profile standard --device cuda --seed 14 --task-split train \
  --calibration runs/cuda-calibration.json
python interaction.py final --session runs/cuda-standard --data data \
  --candidate starter.py --profile standard --device cuda --seed 14 \
  --task-split train --calibration runs/cuda-calibration.json
```

The standard protocol delivers 500 training-step calls, batch 32/context 256,
4,096,000 training tokens and 60 grading batches. It imposes a 300-second training
limit and 600-second per-child watchdog. See the complete
[protocol](../training-env/PROTOCOL.md). A terminal `train` episode grades the
separate development stream; an untouched `heldout` episode grades the final
stream. The shared native accelerator lease covers calibration, training and
grading. Calibration must match source/device/runtime, be fresh and idle, and
pass a live reference/drift check before standard scoring. Do not benchmark speed
tasks beside a learner or bypass a blocked calibration.

Retain the CLI transition, normalized native record, candidate/source/data hashes,
checkpoint and device/runtime identities. Report a blocked/null result as such.
A successful standard native run would establish that workload on that machine;
it would not establish coding-policy learning or an improvement over a matched
initial/no-update baseline. Submitted Python uses local user permissions; the
execution controls are not a hardened sandbox.

## External Prime-RL example: separate two-host, three-GPU plan

The two public configurations are exact copies of the Mac-validated source
examples, with generic `/srv/rsi/` paths and `SCORER_HOST` placeholders:

- [Training configuration](nvidia-example-training-rl.toml)
- [Scoring env-server configuration](nvidia-example-env-server.toml)

They reuse the [existing Verifiers adapter](../suite/integration/README.md) and
an external optimizer. The prepared first topology uses **two hosts and three
physical GPUs**: learner and vLLM policy sampler on separate GPUs of the policy
host; one native scorer GPU on a dedicated scoring host. Scoring concurrency is
one. This is the current planning scope, not a minimum-hardware requirement or a
verified shared-host/single-GPU recipe. Separate hosts avoid nested acquisition
of the same host-wide native lease and reduce scoring contention. The policy
host's sampling/optimization must honor its cooperative lease; native scoring
uses the independent scoring host lease. Other workloads need not honor leases,
so actual idle/resource checks remain necessary.

The example allows at most one outer update, batch 4/group 2, train seeds 14/15,
one terminal code action per episode, 4096 sequence tokens and a 2048-token
completion reservation. It contains no online heldout source. `max_steps=1` does
not bound gathering when episodes are null/invalid or supply no useful reward
signal. An external cumulative wall deadline and owned-process cleanup must be
accepted before execution; no NVIDIA launcher lifecycle has been verified here.

### Pinned upstream sources

Use the exact external checkout and its frozen lockfile/submodules, following
its official setup rather than upgrading the native scorer environment:

- Prime-RL **0.10.0**, commit `e73097df8cd9b9177b84c43a46460f85a0a20157`:
  [setup](https://github.com/PrimeIntellect-ai/prime-rl/blob/e73097df8cd9b9177b84c43a46460f85a0a20157/README.md),
  [dependency declarations](https://github.com/PrimeIntellect-ai/prime-rl/blob/e73097df8cd9b9177b84c43a46460f85a0a20157/pyproject.toml),
  [configuration](https://github.com/PrimeIntellect-ai/prime-rl/blob/e73097df8cd9b9177b84c43a46460f85a0a20157/docs/configuration.md).
- Its editable Verifiers dependency is gitlink commit
  `e888a719adf328c87f271fba39fbc812ca99677b`:
  [source](https://github.com/PrimeIntellect-ai/verifiers/tree/e888a719adf328c87f271fba39fbc812ca99677b).
  The inspected Harness/Agent/Env/Trace/SingleAgentEnv source bytes match the
  installed Verifiers 0.4.0 API. The selected config source package is 0.4.0;
  this is inspected schema/API compatibility, not full ML dependency closure.
- Renderer submodule commit `ca029b524e8d3bb863f0ac30ec46ac16712b9feb`, with
  explicit `prime-qwen3` renderer in the config. See the
  [pinned submodule declarations](https://github.com/PrimeIntellect-ai/prime-rl/blob/e73097df8cd9b9177b84c43a46460f85a0a20157/.gitmodules) and
  [train admission filter](https://github.com/PrimeIntellect-ai/prime-rl/blob/e73097df8cd9b9177b84c43a46460f85a0a20157/src/prime_rl/orchestrator/algo/base.py).

The selected Linux learner stack requires Torch >=2.13.0 and locked vLLM 0.31.0,
with CUDA 13.0 builds. Inspect driver/device support on the actual hosts. Native
scoring keeps its own isolated interpreter and matching requirements/calibration;
`env.agent.harness.python` selects it even when env-server uses the Prime-RL
runtime. Install the shared adapter in the appropriate external environment as
described in its README. These setup steps were not performed on NVIDIA here.

Before using the entry points, replace every path/host placeholder, verify both
actual host/device UUID inventories and resource budgets, and accept a real local
coding-policy checkpoint with the correct renderer. Its complete native prompt
plus reserved 2048 output tokens must fit 4096 without truncation. Bind the actual
prompt/action texts and token IDs to model config, tokenizer assets/chat template,
renderer revision and policy shard hashes. Re-tokenization, model loading,
renderer/vLLM transport and a real adapter episode remain runtime dependencies.
Header/hash consistency alone does not prove a policy can produce useful code.

After those gates and launch/cleanup acceptance, the external entry points are:

```bash
# Dedicated scoring host, from the pinned external Prime-RL environment:
CUDA_VISIBLE_DEVICES=GPU-SCORER uv run env-server @ /path/nvidia-example-env-server.toml

# Policy host: distinct learner and sampler UUIDs, under an accepted bounded launcher:
CUDA_VISIBLE_DEVICES=GPU-LEARNER,GPU-SAMPLER uv run rl @ /path/nvidia-example-training-rl.toml
```

These are prepared entry-point examples, not unattended launch instructions.
`GPU-SCORER`, `GPU-LEARNER`, `GPU-SAMPLER` and `/path/` are placeholders. The
placeholder policy path does not supply model weights; no policy/data downloads
are implicit in the example.

## Evidence required before a learning claim

Only eligible finite terminal standard/train native rewards can feed updates.
Keep smoke/diagnostic, heldout/eval, failed/untrainable, infrastructure/null and
invalid-timing outcomes excluded. Never substitute an empty framework reward sum
of zero for a native measurement or train from a smoke reward. Require real
nonconstant eligible signal for the selected group-based policy objective.

Retain initial/final policy shard and tensor hashes, actual optimizer step count,
checkpoint, consumed native records/rewards and exclusions, prompt/action token
IDs, source/data/config/runtime identities, resource measurements and cleanup.
Saved weight hashes alone prove neither optimizer execution nor improvement.

After stopping the learner, compare the fixed untouched heldout seeds 10014/10015
using initial/no-update and final policies, with identical action budgets,
renderer/tokenizer/resources. Keep every attempted seed, including failures. The
adapter deliberately excludes heldout framework rewards, so comparison must read
independently valid retained native `held_out_loss` records; the corresponding
framework episode may be `ok=False`. Accept a native comparison only when its
framework failure is solely the intentional heldout exclusion. Null, truncated,
missing, source-drift, infrastructure and smoke records remain excluded. Actual
artifact retention and this matched comparison have not executed; ordinary
framework reward evaluation is not a demonstrated substitute. One update cannot
establish general coding gains or recursive improvement.
