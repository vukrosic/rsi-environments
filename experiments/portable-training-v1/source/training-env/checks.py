"""Correctness checks shared by the worker and independent loss evaluator."""
import importlib.util
import sys
import torch
from protocol import VOCAB, MAX_PARAMS, MAX_TENSOR_BYTES

class InvalidCandidate(ValueError):
    pass

def import_candidate(path):
    spec = importlib.util.spec_from_file_location('training_candidate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ('build_model', 'build_optimizer', 'train_step'):
        if not callable(getattr(module, name, None)):
            raise InvalidCandidate(f'missing callable {name}')
    return module

def check_model(model):
    if not isinstance(model, torch.nn.Module):
        raise InvalidCandidate('build_model must return torch.nn.Module')
    # Count all registered parameters, including frozen ones and tied weights once.
    count = sum(p.numel() for p in model.parameters())
    if not 0 < count <= MAX_PARAMS:
        raise InvalidCandidate(f'parameter count {count} outside [1,{MAX_PARAMS}]')
    tensors = list(model.parameters()) + list(model.buffers())
    size = sum(t.numel() * t.element_size() for t in tensors)
    if size > MAX_TENSOR_BYTES:
        raise InvalidCandidate('registered parameter/buffer bytes exceed limit')
    if any(not torch.isfinite(t).all().item() for t in tensors):
        raise InvalidCandidate('nonfinite parameter or buffer')
    return count

def check_logits(logits, x):
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (*x.shape, VOCAB):
        raise InvalidCandidate(f'logits must have shape {(*x.shape, VOCAB)}')
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise InvalidCandidate('logits must be finite floating point')
    return logits

def check_causal(model, x):
    """Reject obvious future-token leakage and nondeterministic inference."""
    model.eval()
    with torch.no_grad():
        a = check_logits(model(x), x)
        b = check_logits(model(x), x)
        if not torch.allclose(a, b, rtol=1e-5, atol=1e-5):
            raise InvalidCandidate('eval logits are nondeterministic')
        for cut in (1, x.shape[1] // 2):
            changed = x.clone()
            changed[:, cut:] = (changed[:, cut:] + 37) % VOCAB
            c = check_logits(model(changed), changed)
            if not torch.allclose(a[:, :cut], c[:, :cut], rtol=1e-4, atol=1e-4):
                raise InvalidCandidate('future tokens affect earlier logits')
            prefix = check_logits(model(x[:, :cut]), x[:, :cut])
            if not torch.allclose(a[:, :cut], prefix, rtol=1e-4, atol=1e-4):
                raise InvalidCandidate('prefix logits depend on sequence length')
    model.train()
