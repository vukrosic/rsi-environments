# Contributing

Start with a CPU smoke task and the commands in [VALIDATION.md](VALIDATION.md).
Use Python 3.12 and install the matching environment's dependencies. Run each
native test folder separately to avoid collisions between development modules.

For a defect report, include the task, source commit, candidate code, Python and
dependency versions, device, seed, profile, and minimal reproduction command.
Remove private paths and credentials from logs before sharing them.

For an optimization result, retain the correctness checks and report measured
workload and timing conditions. Distinguish development/smoke scores from
standard eligible RL rewards. A faster candidate, a trained coding policy, and
a policy that improves on held-out tasks are different results.

Keep contributions focused on environment correctness, reproducibility, task
design, bounded execution, or trainer integration. Do not commit downloaded
data, weights, generated episodes, checkpoints, dependency folders, or caches.
