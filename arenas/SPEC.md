# Arena spec

Every arena follows these rules.

Before building: read [the arena list](README.md) (what exists) and the lessons at the bottom of [BACKLOG.md](BACKLOG.md)
(what went wrong in earlier arenas). Your entry comes from BACKLOG.md.

Reference arena: `env-weight-codec/` (read its README, `weightcodec/league.py`, `weightcodec/sandbox.py`, tests).

## Hard rules

1. **Moving opponent.** Every reward is against the current incumbent of that task (king of the hill) or
   another agent's submission. Never a fixed target or fixed threshold. A submission that beats the incumbent
   by the task's margin becomes the new incumbent; history is kept in `league_state*/` (git-ignored).
2. **No simulated hardware.** No simulated GPU timing, no simulated cache hits, no cost models standing in for
   real hardware. Scores are exact quantities measured on the CPU: counts (accepted tokens, comparators,
   multiplications, clauses, bytes), losses or accuracies of a real model run on the CPU in float64 with a
   fixed seed, or verified correctness. Wall-clock time is only a limit, never the score.
3. **Deterministic.** The same submission scored twice on the same draw gives the same number. Add the
   rerun test: score one submission twice, assert equal.
4. **Held-out draws.** Evaluation instances come from a generator `f(secret salt, task, match seed)`; the salt
   is `<ENV>_SALT` or a git-ignored `.secret_salt`. Public training samples use a public salt.
5. **At least 5 tasks per arena**, each with its own incumbent and history, diverse within the arena.
6. Submission = one Python file the agent writes, run in a separate process via the copied sandbox
   (`env-weight-codec/weightcodec/sandbox.py`, adapt names). Parent does all checking.

## Cheat work: keep it small

Copy the sandbox. Write at most 3 obvious cheat tests (read the answers / patch the scorer / output garbage).
No hack-agent rounds, no long exploit lists. Building more arenas matters more.

## Layout

```
env-<name>/
  README.md        # what the agent writes, task table, reward, limits, commands (same shape as weight-codec)
  <pkg>/           # tasks.py, score.py, league.py, sandbox.py, worker.py, __main__.py
  start/           # a naive starter submission per task (scores > 0, easy to beat)
  tests/           # python3 -m unittest discover -s tests, under 3 minutes total
  league_demo.py   # 2 to 3 rounds: starter -> better submission becomes incumbent
  .gitignore       # league_state*/ league_test/ __pycache__/ .secret_salt hacks/_run/
```

README must state measured numbers: starter score and one improved score per task (headroom shown).

## Resource limits (arenas must run on a laptop CPU)

- Run every test and eval with `nice -n 15` and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- One process at a time per builder, at most 1 GB RAM, test suite under 3 minutes.
- No downloads of models or datasets; generate data or use what is already on disk. No GPU, no torch.mps.
- Tiny models only (numpy, float64, under 100k parameters, under 5 s per evaluation).
- Never start a background process that outlives the command; kill whole process groups.

## Git

One arena per folder; commit only that folder. No Co-Authored-By or AI attribution lines. No absolute home-directory paths,
no company names, no job-search wording in any file. Never touch other folders.
