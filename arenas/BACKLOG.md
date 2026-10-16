# Arena backlog

Arenas to build next. One builder takes one entry. Before starting:

1. Check [the arena list](README.md), so nothing is built twice.
2. Each entry is assigned to one builder at a time ("taken by"); finished arenas move to the arena list.
3. Follow [the spec](SPEC.md) exactly.
4. Read the lessons at the bottom before writing the scorer.

How an entry is written: what the agent writes, what the parent measures, which tasks (at least 5), the starter,
the references (ordered weak to strong), and the checks the parent must do. Builders may change task details if a
task turns out to have no headroom; write down why in the README.

## Training and models

### env-norm-layer
- **Agent writes:** a normalization function and its backward pass, `norm(x, params) -> y` and `grad(...)`. The parent checks both by finite differences.
- **Measured:** held-out loss of a fixed deep numpy MLP or residual net after fixed training.
- **Tasks:**
  - depth 8 / 16 / 32 tanh nets
  - a residual net with no warmup
  - batch size 2 (where batch statistics break)
  - a char LM
- **Starter:** none (the identity).
- **References:** BatchNorm, LayerNorm, RMSNorm, LayerNorm with a learned gain.
- **Taken by:**

### env-activation-fn
- **Agent writes:** an elementwise activation and its derivative, numpy only. The parent checks the derivative by finite differences and checks it is finite on [-50, 50].
- **Measured:** held-out loss of fixed nets after fixed training.
- **Tasks:**
  - spirals
  - parity of 8 bits
  - sine regression
  - a deep narrow net
  - a char LM
  - an extrapolation range
- **Starter:** tanh.
- **References:** ReLU, GELU, SiLU, task-tuned.
- **Taken by:**

### env-regularizer
- **Agent writes:** a penalty term on the weights (with gradient), and optionally a noise injection rule.
- **Measured:** held-out loss with small training sets.
- **Tasks:**
  - 50-sample regression
  - noisy-label classification
  - overparameterized linear model
  - sparse ground truth
  - a char LM with 500 tokens of data
- **Starter:** no penalty.
- **References:** L2 at a fixed value, L1, tuned L2, dropout-like noise.
- **Taken by:**

### env-batch-schedule
- **Agent writes:** the batch size per step under a fixed total sample budget.
- **Measured:** held-out loss.
- **Tasks:**
  - linear regression with two noise levels
  - an MLP on a Gaussian mixture
  - a bigram LM with Adam
  - an MLP with a fixed learning rate
  - a short budget
- **Starter:** constant batch size.
- **References:** linear ramp, step doubling, tuned ramp.
- **Taken by:**

### env-early-stop
- **Agent writes:** a stopping rule that sees the validation curve so far.
- **Measured:** test loss at the step where it stopped, plus a fixed cost per step (state the formula).
- **Tasks:**
  - double-descent curves
  - noisy validation
  - late-improving runs
  - monotone runs
  - runs with spikes
- **Starter:** stop at a fixed step.
- **References:** patience 5, patience tuned, smoothed-curve rule.
- **Taken by:**

### env-model-merge
- **Agent writes:** a merge rule. Given k tiny models fine-tuned from one base (stored, with sha256), it returns one set of weights.
- **Measured:** held-out loss on the union of the k tasks.
- **Tasks:**
  - k = 2 / 4 / 8
  - conflicting tasks
  - one bad model
  - different fine-tune lengths
- **Starter:** the base model.
- **References:**
  - uniform average
  - task arithmetic with a tuned scale
  - TIES-style trim and sign vote
- **Taken by:**

### env-lowrank-adapt
- **Agent writes:** where to put low-rank adapters and with which ranks, under a parameter budget.
- **Measured:** held-out loss after fixed adaptation of a stored base model.
- **Tasks:**
  - a shifted input distribution
  - new classes
  - a changed output scale
  - a char LM on a new alphabet mix
  - a tiny budget
- **Starter:** rank 1 on the last layer only.
- **References:** equal rank everywhere, the most-changed layers first.
- **Taken by:**

### env-test-time-compute
- **Agent writes:** an aggregation policy. Given N samples from a fixed stochastic solver (seeded, built in numpy or pure Python, e.g. a noisy arithmetic solver or a small sampled program search) and an optional noisy verifier with a known error rate, it picks the answer.
- **Measured:** exact accuracy on held-out problems at a fixed sample budget.
- **Tasks:**
  - majority-friendly
  - rare-correct
  - a verifier with 10% / 30% false positives
  - multi-step problems
  - a budget of 4 vs 64
- **Starter:** first sample.
- **References:** majority vote, verifier-filtered vote, weighted vote.
- **Taken by:** in progress (2026-10-16)

### env-reward-data
- **Agent writes:** which preference pairs to label, from a pool, under a budget.
- **Measured:** held-out pairwise accuracy of a fixed tiny reward model trained on the chosen pairs.
- **Tasks:**
  - clean labels
  - noisy annotators
  - a dominant easy cluster
  - near-tie pairs
  - a multi-attribute reward
- **Starter:** random pairs.
- **References:** uncertainty, diversity, uncertainty plus diversity.
- **Taken by:**

### env-embed-compress
- **Agent writes:** a compressor for an embedding table to a byte budget, with a decode function.
- **Measured:** exact recall@10 of nearest-neighbor search with decoded vectors against search with the original ones, on held-out queries.
- **Tasks:**
  - Gaussian clusters
  - a power-law norm distribution
  - low intrinsic dimension
  - high dimension (256)
  - a 4 / 8 / 16 bytes-per-vector budget
- **Starter:** scalar 8-bit truncated to fit.
- **References:**
  - product quantization
  - PCA followed by product quantization
  - residual quantization
- **Taken by:**

### env-grad-compress
- **Agent writes:** a gradient compressor and decompressor for data-parallel training with W workers, run in one process. Error feedback state is allowed.
- **Measured:** held-out loss after fixed training, valid only within an exact bits-per-step budget. Bits are counted exactly from the encoded bytes.
- **Tasks:**
  - linear regression
  - an MLP
  - a char LM
  - 2 vs 16 workers
  - 1 vs 0.1 bits per parameter
- **Starter:** send the top 1% values in float32.
- **References:** sign with error feedback, random-k, top-k with error feedback.
- **Taken by:**

## Algorithms and math

### env-sat-branching
- **Agent writes:** the branching heuristic inside a fixed DPLL solver with unit propagation.
- **Measured:** exact number of decisions to solve held-out instances. A wrong answer is checked against the assignment or a second solver, and is invalid.
- **Tasks:**
  - random 3-SAT near the threshold
  - pigeonhole-style (small)
  - planted solutions
  - graph coloring encodings
  - structured circuit equivalence
- **Starter:** first unassigned variable.
- **References:** most occurrences, Jeroslow-Wang, VSIDS-like activity.
- **Taken by:**

### env-tsp-heuristic
- **Agent writes:** a tour construction plus improvement program, with a time cap.
- **Measured:** exact tour length on held-out instances.
- **Tasks:**
  - uniform random points
  - clustered points
  - a grid with noise
  - non-Euclidean (random metric)
  - asymmetric
- **Starter:** nearest neighbor.
- **References:** 2-opt, Or-opt, 2-opt with restarts.
- **Taken by:**

### env-bin-packing
- **Agent writes:** a packer.
- **Measured:** exact number of bins, then wasted space as the tie-break.
- **Tasks:**
  - uniform sizes
  - many small items
  - triplets
  - 2D rectangles (guillotine)
  - online arrival order
- **Starter:** next fit.
- **References:** first-fit decreasing, best-fit decreasing, a local search.
- **Taken by:**

### env-graph-coloring
- **Agent writes:** a coloring program.
- **Measured:** exact number of colors. An invalid coloring is invalid.
- **Tasks:**
  - random G(n, p)
  - geometric graphs
  - planted k-colorable graphs
  - register-interference-like graphs
  - near-bipartite graphs
- **Starter:** greedy in index order.
- **References:** DSatur, Welsh-Powell, tabu search.
- **Taken by:**

### env-linear-solver
- **Agent writes:** an iterative solver or preconditioner for Ax = b.
- **Measured:** exact count of matrix-vector products to reach a stated residual. Products are counted by the parent's matrix object.
- **Tasks:**
  - a 2D Poisson matrix
  - an ill-conditioned SPD matrix
  - a nonsymmetric matrix
  - a clustered spectrum
  - several right-hand sides
- **Starter:** Jacobi.
- **References:**
  - conjugate gradient
  - conjugate gradient with Jacobi preconditioning
  - GMRES with restart
- **Taken by:**

### env-quadrature
- **Agent writes:** an integration rule.
- **Measured:** exact count of function evaluations to reach a stated error. Evaluations are counted by the parent.
- **Tasks:**
  - smooth functions
  - oscillatory functions
  - endpoint singularity
  - a peak at an unknown location
  - 2D integrals
- **Starter:** trapezoid with doubling.
- **References:** Simpson, Gauss-Legendre, adaptive Gauss-Kronrod.
- **Taken by:**

### env-error-code
- **Agent writes:** a block code and its decoder for a given rate.
- **Measured:** exact decoding error rate on held-out noise draws.
- **Tasks:**
  - binary symmetric channel at p = 0.01 / 0.05
  - erasure channel
  - burst errors
  - rates 1/2 and 3/4
- **Starter:** repetition code.
- **References:**
  - Hamming code
  - random linear code with syndrome decoding
  - interleaving for bursts
- **Taken by:**

### env-proof-search
- **Agent writes:** a search heuristic for a tiny equational or propositional prover. Every step is checked by the parent's kernel.
- **Measured:** exact count of expanded nodes to find a checked proof on held-out theorems, with a node cap.
- **Tasks:**
  - group axioms
  - propositional tautologies
  - list identities
  - arithmetic identities
  - rewriting with lemmas
- **Starter:** breadth-first search.
- **References:** term-size best-first, goal-directed, lemma-reuse.
- **Taken by:**

### env-bandit-policy
- **Agent writes:** an exploration policy.
- **Measured:** exact cumulative regret on held-out bandit instances (seeded rewards).
- **Tasks:**
  - Bernoulli with 2 / 10 / 100 arms
  - Gaussian
  - a slowly drifting best arm
  - contextual linear
  - delayed rewards
- **Starter:** epsilon-greedy 0.1.
- **References:** UCB1, Thompson sampling, LinUCB for the contextual task.
- **Taken by:**

## Systems and data

### env-join-order
- **Agent writes:** a join order (and join method per step) for SQL-like queries over generated tables. The parent executes the plan in pure Python and numpy.
- **Measured:** exact total number of intermediate rows produced. A wrong result is invalid.
- **Tasks:**
  - chain joins
  - star schema
  - skewed keys
  - cyclic joins
  - 8-table queries
- **Starter:** the order as written.
- **References:**
  - smallest table first
  - greedy on estimated cardinality
  - dynamic programming over subsets with exact sample estimates
- **Taken by:**

### env-dedup
- **Agent writes:** a near-duplicate detector for a corpus.
- **Measured:** exact F1 against planted duplicates, at a cap on pairwise comparisons that the parent counts.
- **Tasks:**
  - exact copies with whitespace changes
  - paraphrase-like token swaps
  - code with renamed variables
  - long documents with shared boilerplate
  - multilingual tokens
- **Starter:** exact hash.
- **References:** shingling with MinHash, SimHash, MinHash with LSH bands.
- **Taken by:**

### env-index-structure
- **Agent writes:** a search structure over sorted keys (it may be learned).
- **Measured:** exact count of key comparisons and probes per lookup on held-out queries. A wrong answer is invalid.
- **Tasks:**
  - uniform keys
  - lognormal keys
  - clustered keys
  - keys with heavy duplicates
  - skewed query distribution
- **Starter:** binary search.
- **References:** interpolation search, piecewise linear model with bounded search, B-tree with a tuned fanout.
- **Taken by:**

## Lessons from checking the first 31 arenas

1. **A fixed floor becomes a ceiling.** env-optimizer-rule's valley task let plain momentum reach the 1e-10 loss floor, so no
   submission could ever beat the incumbent. Before committing, check that the best reference is far from any floor or cap:
   - from any floor or clip on the loss
   - from the maximum possible score (spec-decode's n-gram draft reached 8.09 of 9 on one task)
   - from 1.0 accuracy
2. **The starter must be worse than every sensible reference, and should run without diverging.** The lr-schedule starter was worse
   than the seed incumbent on one task. The optimizer-rule valley starter diverged at the shared step size.
3. **Exact-equality tests across BLAS builds fail.** Retraining a stored model gave 7e-14 differences in env-prune-model. Pin
   stored arrays by sha256 and compare rebuilds with a tolerance; keep exact equality only for the score rerun test.
4. **Never trust a gradient or a count the submission reports.** Check gradients by finite differences in the parent, and
   count evaluations, matrix products, comparisons and bytes in the parent's own objects.
5. **Small headroom means slow leagues.** If the best reference is under 10 % better than the starter on a task, change the
   task (harder data, smaller budget) before committing, or say so in the README.
6. **Write "known holes" in every README.** lr-schedule, math-approx, ode-solver and spec-decode still lack that section.
