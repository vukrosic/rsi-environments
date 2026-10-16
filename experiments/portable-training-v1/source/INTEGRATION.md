# Supply an LLM and trainer

The model being evaluated is a fixed 5,063,424-parameter GPT trained on real
TinyStories data for 500 calls at batch 32/context 256. An external LLM proposes
recipes; your external trainer updates that LLM. These are different models.
This source does not offer arbitrary task models or datasets.

Start a fresh interpreter from this source root. Select exact local code paths,
not a private checkout path:

```python
from pathlib import Path
import sys
root = Path.cwd().resolve()
sys.path[:0] = [str(root / 'consumer')]
from native_bridge import load_store, consume_group, recover_quiescent
from journal import Checkpoints, read, sha
```

Supply a fresh external state/output directory, actual trainer bindings and a
durable initial policy/Adam/RNG checkpoint. Keep all run state outside source.
`Checkpoints(trainer_directory, bindings).initialize(...)` must receive real
canonical base/adapter/Adam/RNG hashes and actual state bytes. On resume, reload
the committed `latest.json` state; never initialize over an existing lineage.

Eight actual native rounds must specify the same fixed incumbent, model/data,
runtime and protocol, each with three fresh interaction and three own-validation
seeds. Validation dev windows are disjoint from interaction windows; final
held-out data never supplies a reward. All 48 seeds stay private. A genuine
ten-pair A/A proof tree must derive `max(0.02, 2*spread)`; empty templates and an
unmeasured 2% threshold cannot qualify a group.

```python
checkpoints = Checkpoints(trainer_directory, bindings)
checkpoint, reset_hash = checkpoints.latest()
store = load_store(root, league_state)
obs = store.reset(native_rounds, store.adapter_for(native_rounds), bindings)
assert obs['reset_checkpoint_sha256'] == reset_hash
```

Give the LLM only the public task and `public/PROMPT.txt`. The reference action
is a strict six-field recipe JSON. The pinned compiler converts it to complete
source; `store.measure` takes that source string, not JSON, a path or a claimed
loss. A supplied sampler writes eight ordered completions and their token,
log-probability, compiled source and policy/reset-checkpoint bindings. Then:

```python
samples_path = Path(samples_path).resolve()
samples = read(samples_path)
store.bind_samples(obs['group_id'], {'path': str(samples_path), 'sha256': sha(samples_path)},
                   obs['sampled_policy_sha256'])
for row in samples['samples']:
    assert checkpoints.latest()[1] == reset_hash
    feedback = store.measure(obs['group_id'], row['slot'], row['source_text'])
```

The observation exposes IDs, frozen opponent and sampled-policy hashes, not
private validation inputs. Feedback is status and interaction reward. Valid
reward is incumbent CE divided by incumbent CE plus candidate CE. Different
source ties score 0.5; a verified candidate fault scores zero. Infrastructure,
incomparable results and exact incumbent-source copies are null/excluded. Nulls
are removed before group mean/std; fewer than two samples or zero variance is
a terminal skip without an optimizer step. Never manufacture zero from an error.

Keep one serial on-policy group: reset, sample, measure, consume/save, next reset.
Opponent lag 0/1 may count frozen rewards, but the consumer separately requires
sampled policy tensors to match the latest checkpoint. Lag >=2 is discarded.
Old-policy multi-group updates are outside this implementation's supported scope.

Generation, native evaluation and policy updates occupy separate bounded phases.
Do not hold an outer GPU lease around `measure`: the native evaluator owns its
lease. Before consumption, load the latest committed policy/Adam/RNG and sealed
samples in the admitted outer worker, then hold its existing accelerator lease:

```python
import time
result = consume_group(root, league_state, checkpoints, loaded_update_learner,
    obs['group_id'], deadline_monotonic=min(job_deadline, time.monotonic() + 1200))
```

The bridge wraps the existing `NativeCommitAdapter(TrainerCallback(...))`; no
optimizer or group writer is added. The loaded learner implements
`step(plan, checkpoint, output_dir)`, saves policy/Adam/RNG and returns actual
losses, tensor/component hashes and sampled-log-probability replay evidence.
The native callback returns exactly checkpoint path/hash, group ID, disposition
and policy-state path/hash. It must not load, sample, score, reenter native
methods or acquire another lease. The external supervisor enforces the deadline.

Only durable pointer publication with cumulative `consumed_group_ids` proves
consumption. A flag or unsaved optimizer call does not. Terminal skips persist
`terminal_group_ids`, advance checkpoint index but not optimizer step/adapter/Adam.
Before pointer publication, restore the old state and retry a complete sealed
group once without rescoring. After publication, IDs prevent replay. Incomplete
groups are dropped; broken committed artifacts fail closed. Recovery requires
confirmation the old owned job stopped. Candidate-specific validation and CAS
still decide promotion, only after checkpoint-proven consumption.

The reference learner pins one specific SmolLM2-135M-Instruct setup and helper
functions. Another LLM/trainer requires a separately verified implementation of
the same sampling/checkpoint/receipt contract; generic backend support is not
demonstrated here. The historical helper's unused benchmark CLI is not invoked.
Native submitted Python still has cooperative subprocess containment; this
candidate does not implement the separate UID isolation needed for hostile code.

The 144-episode native group cap plus prerequisite 20-episode A/A is a planned
budget, not observed fit. Native execution, real policy-state restoration,
changed model weights and held-out learning gains remain unmeasured for this
source version. Runtime-only source packaging is not an installed ML package.
