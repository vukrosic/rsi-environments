# Kernel SFT seed

151 JSONL examples across nine fixed workloads. Each row has only `task` and `messages`: one user prompt followed by one assistant reply containing Python code. Message contents retain their exact source UTF-8 bytes. Use the pairs with an existing LLM trainer; no loader framework or model is included.

## Load the conversations

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("kernel_sft_seed.jsonl").read_text().splitlines()]
conversations = [row["messages"] for row in rows]
```

Apply your model's chat template to each conversation and use its assistant
message as the supervised target. The `task` field lets you select workloads.
Model loading, optimization and evaluation belong to your trainer.

## Data and checks

This is seed supervised fine-tuning data, not a trained model or a demonstration of learning. Examples come from repeated optimization attempts, filtered by recorded successful correctness checks and normalized-AST deduplication. These checks were not rerun for this release package. The examples are correlated attempts, not independent samples or a general-correctness guarantee.

Prompts were reconstructed from pinned task/reference/formatter sources, not attested historical agent inputs. **All 151 training prompts retain the pre-patch baseline wording and prompt format with incomplete helper context:** 142 describe PyTorch library baselines; the nine Newton–Schulz prompts say they are timed against the reference itself. Some embedded references call helpers not fully defined in the prompt.

**The c16 evaluation uses patched fastest-baseline prompts with helper definitions, according to the experiment owner.** This is a known train/eval prompt difference, not an identical-prompt setup. This dataset and the [actual c16 result report](../../experiments/kernel-sft-seed/README.md) form one paired data-and-result release. The report finds more replies containing code, without demonstrated correctness improvement or a causal speed gain; the dataset alone establishes no training benefit.

 The [complete reference definitions](https://github.com/vukrosic/rsi-environments/blob/dcecc60d85b82ee35ab7ff6217ab9edb9841e6c0/kernel-timer/tasks.py) include those helpers; their nine reference function ASTs match the pinned reconstruction inputs. Include the helpers when constructing new task prompts. Replies can use reduced internal precision. Inputs are prompt-declared fixed shapes; expected outputs retain the reference dtype/device/shape. All inputs below are float32 except integer labels. Tolerances are the prompt-declared relative-error bounds; the complete error aggregation rule, sampled tensor values, random seeds/distributions and coverage are not supplied here.

| Task | Rows | Declared inputs | Tolerance |
|---|---:|---|---:|
| RMSNorm | 9 | x: 8192×1024; w: 1024 | 0.002 |
| Softmax | 30 | x: 4096×4096 | 0.002 |
| SwiGLU | 20 | a,b: 8192×2816 | 0.002 |
| Cross entropy | 28 | logits: 8192×4096; 8192 int64 labels | 0.002 |
| Causal attention | 8 | q,k,v: batch 8, heads 8, length 1024, head 64 | 0.002 |
| Causal attention backward | 9 | q,k,v,output gradient: same attention shape | 0.002 |
| Newton–Schulz | 9 | 10 matrices of 704×256; five fixed iterations | 0.02 |
| Sliding-window attention | 19 | q,k,v: batch 1, heads 8, length 2048, head 64; window 128 | 0.002 |
| Grouped-query attention | 19 | batch 1, length 1024, head 64; 32 query/8 key-value heads | 0.002 |

Exact remote submitted-source hashes and runtime identity were not attested. Prompts retain their original speed-optimization wording, but this dataset supplies no timing rewards, weights, qualified-RL signal or performance guarantee. Generalization, numerical behavior and any training benefit remain unestablished.

The project-owned seed examples and documentation in this project release are covered by the existing MIT license (copyright 2026 Vuk Rosić), reproduced unchanged in `LICENSE`. This release does not relicense third-party material or replace any applicable third-party rights or notices. Imported libraries are not vendored in this package.
