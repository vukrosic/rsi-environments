"""The reference: mean cross-entropy loss in plain PyTorch, the inputs it is timed on, and the tolerance. Do not edit."""
import torch

TOL = 2e-3  # largest allowed |kernel - reference| / max |reference|


def reference(logits, y):
    m = logits.max(-1, keepdim=True).values
    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
    return (lse - logits.gather(-1, y[:, None]).squeeze(-1)).mean()


def inputs():
    """Timed inputs: logits of 8192 rows by 4096 classes, float32, and 8192 integer labels, on the GPU."""
    return 3 * torch.randn(8192, 4096, device="mps"), torch.randint(0, 4096, (8192,), device="mps")
