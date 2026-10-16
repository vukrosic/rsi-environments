# Training recipe groups — public-source candidate

This is an unexecuted source successor for researchers supplying an LLM and
trainer. It grades recipes for a fixed small GPT/TinyStories task. It is not
an installed package, arbitrary-model/data API or one-command model reproduction.
No new training, promotion or learning gain is demonstrated by this candidate.

The native contract is `training-group8-public-v1`; its runtime-only manifest
has 20 Python files. The consumer has its own source version and requires that
exact new native manifest. Earlier software results belong to earlier source;
new composition/import/packaging controls must run before claiming compatibility.

After an allowed packager has actually assembled this candidate, from its root:

```bash
python3 -I -S -B researcher.py
python3 -I -S -B researcher.py --metadata-imports
```

The first command checks the export file map. The second also imports metadata
interfaces and verifies module origins without importing model libraries. Neither
runs models, native scoring, an optimizer, downloads, tests or a provider client.
These commands have not yet been executed for this source version.

Layout: `training-env/` and `suite/` are native runtime code; `consumer/` is the
existing checkpoint/transaction implementation with new source pins;
`framework/recipe/` contains only the compiler/verification utilities and their
original source/model manifests; `framework/grpo-helpers/` contains the explicitly
pinned helper modules. Their old benchmark CLIs are not supported entry points.
Model weights, prepared data, calibration, A/A receipts, split seeds, checkpoints,
host details and real run histories are excluded.

See [the integration sequence](INTEGRATION.md) and [public recipe prompt](public/PROMPT.txt).
Python 3.11+ and Linux are the source target. Native model execution additionally
needs an explicitly verified Torch/NumPy/tokenizer runtime; the reference
consumer also needs its supported Transformers/model assets. Nothing installs
or substitutes those dependencies here.
