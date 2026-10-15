"""Check that kernel() gives the reference's answer on one set of inputs. No timing. Use it as often as you like."""
import torch
from kernel import kernel
from reference import TOL, inputs, reference

with torch.no_grad():
    x = inputs()
    want, got = reference(*x), kernel(*x)
    err = (got.float() - want).abs().max().item() / want.abs().max().item()
print(f"{'OK' if got.shape == want.shape and err <= TOL else 'WRONG'}: relative error {err:.2e} (allowed {TOL:.0e})")
