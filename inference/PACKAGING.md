# Versioned inference package

The installable package is `vuk-rsi-inference` version `0.2.0`. It supports
CPython 3.12 on macOS with CPU inference or Apple MPS. The wheel contains the
evaluator source, task metadata, and the shared MIT helpers. It does not contain
model weights, generated episodes, evidence logs, or local caches. The pinned
SmolLM2-135M files are fetched separately into the Hugging Face cache.

From this directory, build and inspect a wheel without modifying `.venv/`:

```bash
python3.12 -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/rsi-wheel .
python3.12 -m zipfile -l /tmp/rsi-wheel/vuk_rsi_inference-0.2.0-py3-none-any.whl
```

Install it into a clean Python 3.12 environment with the pinned runtime
dependencies listed in wheel metadata:

```bash
python3.12 -m venv /tmp/rsi-package-env
/tmp/rsi-package-env/bin/python -m pip install /tmp/rsi-wheel/vuk_rsi_inference-0.2.0-py3-none-any.whl
/tmp/rsi-package-env/bin/rsi-inference --version
/tmp/rsi-package-env/bin/rsi-inference generate-assets
```

The asset command downloads the pinned public model revision. Scoring needs that
model in the local cache. `rsi-inference score candidate.py --episode runs/demo
--mode check --diagnostic` verifies a candidate without a timing reward, and
`rsi-inference replay RUN_DIR` checks a saved receipt without model execution.
The Python interface is `from inference_env import CodingAgentEnvironment,
InferenceEnvironment`. A reset creates a new local episode directory; actions
are recorded there. A syntactically invalid terminal action is rejected before
any model work.

The wheel installs evaluator modules below `inference_env._runtime`; it does not
install generic top-level modules such as `scorer` or `assets`. The scorer still
copies script-mode snapshots into each run directory for durable receipts.
Execution of submitted candidate code is trusted local Python code, not a
hostile-code sandbox. Do not use this package for untrusted submissions.
