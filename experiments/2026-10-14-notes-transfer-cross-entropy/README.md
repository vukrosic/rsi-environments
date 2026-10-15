# Do attention notes help on a different kernel? cross_entropy (2026-10-14)

4 Codex agents got the notes round 3 wrote on causal attention ([notes-r3.md](../2026-10-14-notes-attention/notes-r3.md)),
4 got the plain prompt, all running at the same time on an M4 Mac. Task: fused cross-entropy loss. Unchanged code:
18.5 ms ([base.json](base.json)).

| Prompt | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Median final | Tokens, median |
|---|---|---|---|---|---|---|
| plain + attention notes (`transfer-*`) | 2.29 | 1.98 | 1.76 | 2.32 | 2.13 | 48.8k |
| plain (`control-*`) | 1.73 | 2.68 | 2.65 | 2.16 | 2.41 | 31.6k |

Pass rule, fixed before the batch: the worst notes agent beats the median plain agent. 2.32 < 2.41 passes on paper,
but the result can't be read:

- All 8 agents wrote a fused Metal shader on their first bench run (plain first runs 1.58–2.52 ms, 7–12× faster
  than unchanged). The notes had nothing left to teach.
- Two video renders ran on the same GPU. One agent's own runs moved between 1.6 and 4.5 ms, more than the gap
  between the groups.
- The notes cost about 17k more tokens per agent.

## Prompts

- plain: "Make kernel() as fast as you can."
- `transfer-*`: the same, plus: "LESSONS.md in this folder holds notes from earlier agents who optimized a different
  kernel (causal attention) on this same Mac. Read it before your first bench run."

## Files

- `start/`: the starting folder every agent got (`kernel.py`, `reference.py`, `bench.py`, `check.py`).
- `agents/<arm>-<n>/`: `start.py`, `run1.py`…`run10.py` (the kernel at each bench run), `final.py`, `results.jsonl`
  (every timing; the last line is the final, 5 timings on unseen inputs, median), `TRACE.md` (the agent's messages
  and code changes per run), `LESSONS_given.md`, `LESSONS_written.md`. Raw transcripts are not published.
- Harness: [../2026-10-14-notes-attention/harness/](../2026-10-14-notes-attention/harness/) (`kloop.py --kernel cross_entropy`).
