# Arenas

Arenas are RL environments where a coding agent writes one program and is scored against the current best
submission for each task (king of the hill), never against a fixed target. Every score is an exact quantity
measured on the CPU. **The arena code is not in this repository yet;** this page lists what exists and its status.

Every arena follows the [arena spec](SPEC.md). Arenas planned next, with lessons from checking the first ones: [BACKLOG.md](BACKLOG.md).

Status: **real** = exact CPU score, tests pass; **building** = in progress; **parked** = scores a
simulated GPU, kept for a rebuild on a real GPU (no new simulated arenas).

## Training and models (agent writes part of a training run, score = held-out loss of a tiny numpy model)

| arena | the agent writes | status |
|---|---|---|
| env-optimizer-rule | the optimizer update rule | real |
| env-lr-schedule | the learning-rate schedule lr(t, T) | real |
| env-init-scheme | the weight initialization | real |
| env-curriculum | the order of training examples | real |
| env-data-mixture | the mixture weights over data sources | real |
| env-data-select | the training subset at a fixed size | real |
| env-tiny-arch | the architecture under 5,000 parameters | real |
| env-prune-model | which weights to prune | real |
| env-quant-model | the quantizer at a fixed bit budget | real |
| env-weight-codec | a lossy codec for weight tensors | real |
| env-spec-decode | the draft policy for speculative decoding | real |
| env-tokenizer-league | the tokenizer trainer | real |
| env-attn-pattern | the sparse attention mask under a budget | building |
| env-loss-design | the loss function and its gradient | building |
| env-augment-policy | the data augmentation | building |
| env-distill-target | the targets built from teacher logits | building |
| env-pos-encoding | the positional encoding, scored on longer sequences | real |
| env-moe-router | the mixture-of-experts router | real |
| env-active-learning | which pool examples to label next | real |
| env-test-time-compute | the answer-aggregation policy over N sampled solutions (accuracy at a fixed sample budget) | real |

## Algorithms and math (score = an exact count or exact error)

| arena | the agent writes | status |
|---|---|---|
| env-matmul-rank | fast matrix multiplication schemes (multiplication count) | real |
| env-sorting-network | sorting networks (comparator count) | real |
| env-circuit-min | Boolean circuits (gate count) | real |
| env-superopt | straight-line programs in a tiny ISA (instruction count) | real |
| env-expr-rewrite | an expression simplifier (node count) | real |
| env-math-approx | approximations of exp, log, sin, tanh, erf, rsqrt (error at a fixed cost) | real |
| env-ode-solver | an ODE integrator (RHS evaluations to reach an error) | real |
| env-maxsat-heuristic | a MaxSAT heuristic (satisfied weight at a fixed query count) | real |
| env-hash-design | a hash function (collisions and maximum bucket load) | real |
| env-regex-synth | regex synthesis from examples (held-out accuracy, then length) | real |
| env-symbolic-regression | formula search (extrapolation error plus a size penalty) | real |

## Systems and data (score = exact bytes, moves or ranking quality)

| arena | the agent writes | status |
|---|---|---|
| env-mem-planner | a static activation memory planner (peak bytes) | real |
| env-text-codec | a lossless text and structured-data codec (compression ratio) | real |
| env-retrieval-rank | a search ranker (nDCG@10) | real |
| env-reg-alloc | a register allocator (executed loads, stores and moves) | real |
| env-graph-partition | a balanced graph partitioner (cut weight) | real |
| env-game-league | game players for 6 board games (win rate against the king) | real |

## Parked (simulated GPU, rebuild on a real GPU)

env-kv-evict, env-cluster-sched, env-pipeline-place.
