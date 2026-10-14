# Reproducible training data environment

Create an isolated environment on another machine:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Linux or macOS
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt  # exact versions verified locally
```

The preparation commands do not download data. Obtain the TinyStoriesV2-GPT4
train and validation text files from the dataset publisher's official
[Hugging Face dataset repository](https://huggingface.co/datasets/roneneldan/TinyStories):

```bash
mkdir -p source
curl -L -o source/TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -r 0-99999999 -o source/TinyStoriesV2-GPT4-train-100mb.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
python prepare.py --train-text source/TinyStoriesV2-GPT4-train-100mb.txt \
  --valid-text source/TinyStoriesV2-GPT4-valid.txt --output data
```

This fits the 4,096-token byte-level BPE on training stories only, then assigns
complete validation stories to separate `dev` and `final` splits. The output
directory contains `train.npy`, `dev.npy`, `final.npy`, `tok.json`, and a
SHA-256 manifest. Keep the downloaded raw text out of source control. Before
redistributing data or prepared assets, review the dataset's current license
and attribution terms and follow them.

To reuse an existing preparation without changing its files:

```bash
python prepare.py --from-existing /path/to/old-data --output data
```

That directory must contain `train.npy`, `valid.npy`, and `tok.json`; both
token streams must end at complete story boundaries marked by the tokenizer's
`<|endoftext|>` token. The old validation stories are divided at a story
boundary near the token-count midpoint.

For plumbing checks only, create deterministic synthetic data explicitly:

```bash
python prepare.py --synthetic --output smoke-data
```

Synthetic manifests are labeled `kind: synthetic` and are rejected by the
standard training profile. Load one only with `load_data("smoke-data",
profile="smoke")`. They are not a substitute for TinyStories.

Keep this folder alongside `../suite/` (shared helpers). Copying just this
folder without the suite is incomplete. Run CPU tests with:

```bash
python -m unittest discover -s tests -v
```

`requirements.txt` gives the supported minor-version ranges; `requirements-tested.txt`
pins the tested stack. No fresh internet installation on a second machine has been
verified in this build. See [RL-INTEGRATION.md](RL-INTEGRATION.md) for the policy
trainer contract. The TinyStories dataset card currently labels its license
`cdla-sharing-1.0` (checked October 14, 2026); source data is not included.

Preparation now refuses any existing output directory to preserve data/receipts.
Choose a fresh destination for a rebuild. `reproduction_check.py` can extract a
source-only package and test it using an explicit fresh-venv Python. It preserves
venv executable paths, clears inherited Python import paths, validates archive
members, performs explicit synthetic preparation, rejects synthetic standard
mode and runs a complete CPU terminal episode. See the current changelog for
the measured same-Mac installation/reproduction result.
