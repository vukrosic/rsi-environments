# Experimental portable training source

This branch previews a source bundle for Training Improvement and its existing
trainer bridge. An LLM proposes a training recipe; the task trains a separate
model and compares its loss with a frozen incumbent. The fixed reference is
reported separately. This source version is experimental.

The [source guide](source/README.md) and [integration contract](source/INTEGRATION.md)
describe the task, required assets, budgets, eligibility and trainer responsibilities.
The bundle currently fixes the task model and data. Another task model or trainer
requires separately verified bindings.

The workflow checks exact source bytes and imports the metadata modules on a
standard Linux runner. It uses only Python's standard library, with no model,
dataset, ML dependency installation, GPU, task episode or optimizer update.
Passing it establishes only those source-loading checks. It does not verify
native training, policy restoration, valid GPU rewards or learning gains.

`source/EXPORT-MANIFEST.json` lists the original allowlisted source identities.
Its historical `private_candidate` field describes the prepared candidate and
is preserved with that source version; this experimental branch is public.
No runtime state, private seed plans, checkpoints or raw execution logs are
included. The source payload is separate from the existing released examples.

This is not a new release tag. Read the workflow result for this exact branch
commit before claiming a source-loading pass. Full dependency setup and an
actual supported-GPU episode remain separate release requirements.
