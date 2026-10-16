# Kernel SFT seed experiment: more code replies, no demonstrated correctness gain

Fine-tuning Qwen2.5-Coder-1.5B-Instruct on the paired 151-example seed data
increased replies containing code. It did **not demonstrate improved
correctness**, and these results establish **no causal speed gain**.
The [dataset](../../datasets/kernel-sft-seed/README.md) and this actual-result
report form one paired data-and-result release.

| Pooled outcome | Base | SFT | Two-sided Fisher p, unadjusted |
|---|---:|---:|---:|
| Correct, nine training tasks | 16/288 | 24/288 | 0.25 |
| Correct, three held-out tasks | 5/96 | 2/96 | 0.44 |
| Replies containing code, training tasks | 214/288 | 265/288 | 1.4e-8 |
| Replies containing code, held-out tasks | 67/96 | 89/96 | 6.9e-5 |

All sampled replies form the denominators, including replies without code.
“Contains code” is the evaluator's saved `has_code` count; it does not mean the
code imports, computes correct values or runs faster. A non-significant
correctness comparison does not prove equal performance. The lower held-out
count does not establish general harm.

## Per-seed results and task audit

| Condition | Evaluation seed label | Correct, training | Has code, training | Correct, held out | Has code, held out |
|---|---:|---:|---:|---:|---:|
| base | 1 | 7/144 | 98/144 | 2/48 | 32/48 |
| base | 2 | 9/144 | 116/144 | 3/48 | 35/48 |
| sft | 1 | 14/144 | 127/144 | 1/48 | 48/48 |
| sft | 2 | 10/144 | 138/144 | 1/48 | 41/48 |

[RESULTS.json](RESULTS.json) preserves every returned per-task sample,
code-presence and correctness count for all four condition/seed summaries,
plus their reported selected-best timing fields and original artifact hashes.
Seeds 1 and 2 are labels from the returned filenames/owner description; the
summary JSON does not itself contain a seed field.

Each condition/seed summary specifies 16 replies per task at temperature 0.8,
five timing trials and 15 retrials for a selected best sample. The fixed task
set is tiny: nine training workloads and three held-out workloads
(decode attention, selective scan and int4 matmul). These are two evaluation
sampling seeds for one reported SFT adapter, not two training replications.
Raw generated samples, per-sample verdicts and timer/source bindings were not
returned, so aggregate counts cannot establish broader numerical or speed
behavior. Returned aggregates contain no per-sample failure categories, so infrastructure errors cannot be separated from unsuccessful code in these counts.
The retained best timing ratios are descriptive selected results,
not qualified RL speed rewards or evidence against the strongest baseline.

## Training and data identity

The experiment owner reports LoRA rank 16 on q/v projections, 57 optimizer
steps and three epochs. The returned training record also reports rank 16,
three epochs, 151 trained rows, no skipped rows, accumulation 8, learning rate
0.0002, maximum sequence length 2048, training seed 0 and bfloat16. The step
count and q/v target modules are owner-reported configuration; no optimizer
step or loss log was returned to independently verify them.

The returned record identifies training input SHA256
`6455bd80e095e0b201d0cc8481dee7802ae2cf220b56b09ad0abbfeb212148ce`.
The paired public JSONL is the sanitized task/messages projection, SHA256
`60c5ec0ba86507951c1e53c69a42e0111010073d246bf175e3f546f52dc60bcd`.
These container hashes differ; all 151 task and prompt/reply strings were
checked for exact correspondence. The original data and public projection
were preserved, not rewritten for this report.

Actual returned base-start/base-end and adapter-end digest strings are in
RESULTS.json. The two base digest strings agree. No adapter-start digest,
checkpoint bytes or loss trace was returned; the metadata is not independent
weight-change or loss-improvement proof. No unreturned checkpoint hash is added.
The experiment owner's exact-run source/mutation check report is still pending;
no passing source/mutation acceptance is inferred from these summaries.

## Prompt difference and statistical limits

Training prompts were reconstructed from pinned pre-patch task/formatter
sources, rather than attested historical agent inputs. They retain older
library/reference-baseline wording and incomplete helper context. The owner
reports that evaluation used patched fastest-baseline prompts with helper
definitions. Actual evaluation prompt text/source identity is not contained
in these four summaries. This is a known training/evaluation prompt difference,
not an identical-prompt setup. Base and SFT nominal evaluation settings match.

The p-values are two-sided Fisher exact calculations from the pooled aggregate
success/failure tables, reproduced with integer-combination arithmetic only.
They condition on the margins and treat replies as independent Bernoulli
outcomes. Shared tasks, prompts, adapter and sampling seeds introduce clustering
that this pooled calculation does not model. The four tests are unadjusted
for multiple comparisons and are not paired or task-stratified. They describe
this small fixed evaluation and do not establish generalization or a causal
speed advantage. The experiment was not rerun while preparing this release.
