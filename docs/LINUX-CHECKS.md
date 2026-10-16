# Bounded Linux source and package checks

The `Linux source and package checks` workflow runs on one standard GitHub-hosted
`ubuntu-24.04` runner with Python 3.12. It is a CPU software check of the public
checkout. [View runs](https://github.com/vukrosic/rsi-environments/actions/workflows/linux-checks.yml). It does not qualify the inference runtime for Linux, qualify a GPU
backend, measure optimization speed, or establish RL reward validity.

The inference package currently documents macOS CPU/MPS runtime support. Here,
`pip install --no-deps --no-build-isolation` deliberately checks only its public
source build/install, namespace, metadata and dependency-free CLI surface. Full
ML dependency installation and the lazy `CodingAgentEnvironment` and
`InferenceEnvironment` imports are outside this check.

## Finite checks

The workflow installs only `pytest==8.4.2` and its small Python dependencies in
the control environment. It runs these existing suites, in separate invocations
where flat source modules need their own working directory:

| Working directory | Command after `python -m` | Evidence on success |
| --- | --- | --- |
| Repository root | `pytest -q suite/tests/test_shared.py kernel-env/tests/test_packaging.py` | Reward/record validation, detached atomic records, cooperative cross-process lease release, and allowlisted/reproducible kernel source archives |
| `inference` | `pytest -q tests/test_package_layout.py` | Wheel contents, private namespace rewriting, no weights/run evidence in the wheel, and reproducible wheel bytes |
| `training-env` | `pytest -q tests/test_process_control.py tests/test_reproduction_package.py` | Owned child cleanup with an unrelated process preserved; archive path/type/size/member limits and expired reproduction deadlines |

The process-control suite launches only its own short-lived Python fixtures. It
does not run agent submissions. Kernel packaging fixtures are synthetic source
trees. The inference layout check builds a wheel from the public source.

A second fresh virtual environment receives the actual inference source through:

```sh
python -m pip install --no-deps --no-build-isolation "$GITHUB_WORKSPACE/inference"
```

From outside the checkout, isolated Python (`-I`) checks the installed package
path, package/distribution version agreement, `inference_env.cli.main`, and the
installed `rsi-inference --version` / `--help` entry point. It also checks that
Torch and generic top-level `scorer`, `assets`, and `worker` modules were not
installed. This prevents a checkout import from masquerading as install success.

The training interaction API imports Torch transitively. The integration
protocol tests import the framework package, which initializes Verifiers and
OpenAI dependencies. These are deliberately excluded from this small job rather
than stubbed or reported as tested.

## Resource and execution boundaries

- One job, no matrix, a 15-minute whole-job deadline, and a 120-second limit on
  each installation or test command. The final import/CLI check has 30 seconds.
  Step deadlines and `timeout` kill grace periods provide additional bounds.
- No model or dataset downloads, Torch installation, training, candidate scoring,
  GPU use, cloud-provider calls, or local Mac execution. Hugging Face offline
  variables are set as an extra guard, not as a network sandbox.
- Standard public GitHub-hosted runner only; no self-hosted or larger runners,
  secrets, caches, or artifact uploads. Evidence is the workflow's ordinary logs
  and exit status, attached by GitHub to the checked-out commit.
- Read-only repository permissions and a shallow checkout without persisted
  credentials or tag fetching. The workflow does not commit, publish, push, or
  mutate tags. Official actions are pinned to commit SHAs.
- Manual dispatch is accepted only for `main`. Scoped pushes to `main` rerun the
  job when checked sources or this workflow change. There is no pull-request,
  schedule, or `pull_request_target` trigger. A newer run cancels an older run
  for the same ref.

Adding this workflow does not establish a passing Linux result. A successful
GitHub run at a specific commit is the evidence; a prepared YAML file or local
source inspection is not. A red run must remain a failure until investigated.

## First hosted result

[Run 38856915979](https://github.com/vukrosic/rsi-environments/actions/runs/38856915979) passed on Ubuntu 24.04 / Python 3.12.15 at commit `751fc99c7897e987b8f639a969e067508e638ccb`. The job completed in 17 seconds, including the source build/install and isolated CLI checks above. This establishes only the listed software behavior. Subsequent commits have their own workflow results.
