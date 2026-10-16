# One-command release check

From a clean checkout:

```sh
python release/check.py
```

Standard library only (Python 3.11+). Add `--list` to print the checks without running them, `-v` to show the output tail of passing checks too.

## Output

One line per check, then a summary:

```
PASS  unit: kernel-env       <pytest summary> (12.3s)
SKIP  timer hacks: ...       needs CUDA GPU (macOS has no CUDA)
FAIL  unit: inference        exit 1, 8.0s
      <last lines of the failing output>

9 passed, 1 failed, 3 skipped
```

| Word | Meaning |
|---|---|
| `PASS` | The check ran and succeeded. |
| `FAIL` | The check ran and failed, timed out, or an expected evidence file is missing. The last lines of its output are printed under it. |
| `SKIP` | The check cannot run on this machine. The reason follows: a missing Python module (`pytest`, `torch`, `numpy`, `verifiers`) or no CUDA GPU. A skip is not a pass. |

Exit code 0 only if no line says `FAIL`. A run with skips prints `no failures; skipped checks did not run on this machine`.

## What is checked

| Check | What runs | Needs |
|---|---|---|
| `unit: shared records + leases` | `suite/tests` | pytest |
| `unit: kernel-env` | `kernel-env/tests` | pytest, torch, numpy |
| `unit: training-env` | `training-env/tests` | pytest, torch, numpy |
| `unit: inference` | `inference/tests` | pytest, torch, numpy |
| `unit: framework adapter` | `suite/integration/tests` | pytest, verifiers |
| `unit: kernel-timer report tool` | `kernel-timer/tools/test_aa_report.py` (run from `kernel-timer`) | pytest |
| `unit: private-wording scanner` | `tools/test_check_private_words.py` | nothing extra |
| `hygiene: no private wording in tracked files` | `tools/check_private_words.py` | git |
| `timer hacks: every known fake scores no gain` | `kernel-timer/hacks/run.py attention`: each known fake must score 0 or no gain | **CUDA GPU**, torch |
| `evidence files: test 1` to `test 5` | the files that back each release test exist in this repository | nothing |

Unit tests run with `CUDA_VISIBLE_DEVICES` empty. The suites mock their device checks; none starts GPU work. Each suite has a time limit and is killed with its child processes when it expires.

## What needs a GPU

Only the timer-hacks check. It needs an NVIDIA GPU that is otherwise idle, because it times kernels. Without CUDA it prints `SKIP ... needs CUDA GPU`. It never uses an Apple GPU. Expect up to an hour on a GPU (about 20 fakes, 5 timed trials each). The hacks runner exits 1 if any fake got a gain and 2 if a fake could not be checked; both print `FAIL`.

## Limits

The evidence-file lines check that the files exist, not that their numbers meet the release thresholds. Test 1 (same-task gain), test 2 (transfer) and test 3 (at least 8 tasks with a 1.5x gap) are read from the experiment folders and `kernel-timer/VALIDATION-V5.json`, and need a GPU to reproduce.
