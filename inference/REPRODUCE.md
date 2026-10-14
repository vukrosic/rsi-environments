# Install and reproduce

Tested installation target: Apple M4 (`Mac16,1`), 16 GB unified memory, macOS,
CPython 3.12.8. The clean `.venv` was installed from `requirements.txt`; its exact
transitive packages are in `requirements-lock.txt`. No hosted inference/API key
or paid resource is needed. Model download is about 273 MB; float32 loading uses
more memory than its stored checkpoint. Preserve model assets in the HF cache.

```bash
cd inference
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python assets.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scorer.py solution.py --episode episodes/baseline --mode check --diagnostic
```

`assets.py` fetches the fixed commit, never floating `main`, and writes the asset
manifest. Keep the supplied manifest for comparison; fetching again should give
identical hashes. Scoring is offline and verifies every asset before execution.
No sibling environment is imported when this directory is copied alone: the
shared helpers fall back to the bundled `shared/` copies, retaining their MIT notice.

After inspecting machine load and when competing work has ended:

```bash
.venv/bin/python scorer.py solution.py --episode episodes/cpu-baseline --device cpu
.venv/bin/python scorer.py attempt.py --episode episodes/cpu-fp16 --device cpu
.venv/bin/python scorer.py solution.py --episode episodes/gpu-baseline --device mps
.venv/bin/python scorer.py solution.py --episode episodes/gpu-baseline --device mps --mode final
```

For correctness when busy, add `--diagnostic`; normalized timing reward becomes
null. The fp16 attempt is a documented failed optimization under the fixed tolerance,
not an expected speed win. Run an unchanged submission multiple times to quantify
noise before claiming another optimization succeeds. Never compare CPU and MPS
numbers as though they were the same condition.

Minimal research loop, from this directory:

```python
from interface import InferenceEnvironment
env = InferenceEnvironment('episodes/research-loop', device='cpu')
feedback = env.step('solution.py', diagnostic=True)
print(feedback['status'], feedback['reward'], feedback['metrics'])
# Edit solution.py; repeat env.step(...) within the ten-call budget.
final = env.finalize('solution.py', diagnostic=True)
```

The last file is what finalization freezes; best historical feedback is not selected
automatically. Each call writes a distinct durable directory. To replay a stored
case, run its copied worker with its stored inputs through an evaluator-owned
process under the same accelerator lease; that is reproduction, not a fresh final
score. `.pt` inputs/outputs are loaded with `weights_only=True`. `result.json` records
the exact interpreter, dependency versions, source hashes and measured conditions.

### Check saved outputs without rerunning the model

```bash
.venv/bin/python replay.py episodes/release-baseline/final-01-61e17f7f --output replay-report.json
```

Supply the complete run directory containing its raw `.pt` tensors, not the
lightweight evidence copy. This verifies recorded source/evaluator/input/output
hashes and rechecks oracle consistency plus candidate correctness on CPU. It does
not execute model code, measure latency, or emit a fresh training reward: replay
always returns `reward: null`. Modified or missing artifacts return an error.
Saved oracle disagreement takes precedence over candidate correctness. This is
local independent checking of a receipt, not independent reproduction of timing.

For a standalone copy, copy this directory excluding `.venv/`, `__pycache__/`, and
local episodes, then repeat installation and `assets.py`. Keep `requirements-lock.txt`,
the manifest, and `shared/`. The bundled helpers use the suite's same default lock;
do not set a different lock path while other local environments are using it.
