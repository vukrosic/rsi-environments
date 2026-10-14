# Apple GPU check — v0.1.0-alpha.3

On October 14, 2026, one real TinyStories inner-GPT smoke completed on Apple MPS. The model had 5,063,424 parameters. The fixed smoke protocol scheduled 3 training calls, of which 2 had positive learning rate; it processed 96 training tokens and graded 64 development tokens. The terminal development loss was 8.111218929290771. A smoke reward was recorded for plumbing only and is excluded from policy training.

An independent review verified the frozen candidate's observation wrappers, checked that the original starter functions and their arguments and returns were preserved, and rechecked source/data identities, initial and final state hashes, and the native receipts. It observed MPS-resident inputs, parameters, gradients, optimizer moments and logits at the audited boundaries, including the independent grader. This supports functional MPS execution at those tensor boundaries. It is not a per-operator GPU-only trace: CPU metadata, preprocessing, checkpoint storage, synchronization and hash instrumentation also participate.

## Reproduce the workload

Follow [training-env/INSTALL.md](../training-env/INSTALL.md) to create the environment and prepare the real TinyStories data. On an Apple Silicon Mac with a PyTorch build that provides MPS, run from `training-env/`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python evaluate.py --output runs/apple-mps-smoke --data data \
  --candidate starter.py --profile smoke --device mps --split dev --seed 43
```

This runs the unchanged public starter with the same fixed smoke protocol and seed. It does not recreate the privately reviewed observation wrappers, tensor-boundary audit, or exact recorded hashes; the CLI reports its own result. Smoke results are not eligible for standard RL training.

## Scope

| Environment | Current accelerator statement |
|---|---|
| Inference optimization | macOS CPU and Apple MPS are the supported runtime; CUDA is unsupported. |
| Training improvement | One real-data smoke completed on Apple MPS. CUDA was not executed, so compatibility is unverified. |
| Kernel optimization | Selected tasks include Apple/Metal paths. CUDA execution is conditional on task/backend support and was not executed in this release. |

No initial development loss was measured, so the changed inner-model weights do not establish a learning gain. No standard-budget CUDA execution, coding-agent weight update, learned policy improvement, or recursive improvement was measured. The captured MPS allocation samples are not peak physical GPU memory, and observed runtime is not a calibrated performance result.

See [apple-gpu-check.json](apple-gpu-check.json) for sanitized protocol, metric, device, and hash evidence.
