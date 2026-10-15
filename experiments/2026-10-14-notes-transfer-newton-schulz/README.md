# Do attention notes help on a different kernel? newton_schulz (2026-10-14)

Three groups of 4 Codex agents, all running at the same time on an M4 Mac. Task: the Newton-Schulz iteration
Muon uses (5 rounds of matrix products). Unchanged code: 8.9 ms ([base.json](base.json)). PyTorch's fused
multiply-add version is 1.45× faster, fp16 1.71×.

| Prompt | Finals (ms) | Median | Worst | First run, median | Tokens, median |
|---|---|---|---|---|---|
| attention notes (`transfer-*`) | 1.56, 1.84, 2.51, 2.70 | 2.18 | 2.70 | 6.9 | 98k |
| generic advice (`advice-*`) | 2.28, 2.35, 2.37, 2.54 | 2.36 | 2.54 | 4.8 | 86k |
| plain (`control-*`) | 1.77, 2.27, 2.40, 2.80 | 2.33 | 2.80 | 5.1 | 78k |

Pass rule, fixed before the batch: the worst notes agent beats the median plain agent. 2.70 > 2.33: **no transfer**.

- The generic advice ([advice-generic.md](advice-generic.md), 1,190 words) is a control of about the same length as
  the notes (1,105 words), written from no agent's runs.
- The notes gave the two fastest agents and the slowest first runs: those agents wrote custom tiled shaders first,
  as in attention, before tuning them.
- All 12 finals use a Metal shader with fp16. Most replace some of the 5 iterations with one polynomial of the Gram
  matrix, which the 2e-2 tolerance allows.
- Video renders ran on the same GPU during the batch.

## Prompts

- plain: "Make kernel() as fast as you can."
- `transfer-*`: the same, plus: "LESSONS.md in this folder holds notes from earlier agents who optimized a different
  kernel (causal attention) on this same Mac. Read it before your first bench run."
- `advice-*`: the same, plus: "LESSONS.md in this folder holds general advice on optimizing GPU kernels on this Mac.
  Read it before your first bench run."

## Files

- `start/`: the starting folder every agent got (`kernel.py`, `reference.py`, `bench.py`, `check.py`).
- `agents/<arm>-<n>/`: `start.py`, `run1.py`…`run10.py` (the kernel at each bench run), `final.py`, `results.jsonl`
  (every timing; the last line is the final, 5 timings on unseen inputs, median), `TRACE.md` (the agent's messages
  and code changes per run), `LESSONS_given.md`, `LESSONS_written.md`. Raw transcripts are not published.
- Harness: [../2026-10-14-notes-attention/harness/](../2026-10-14-notes-attention/harness/) (`kloop.py --kernel newton_schulz`).
