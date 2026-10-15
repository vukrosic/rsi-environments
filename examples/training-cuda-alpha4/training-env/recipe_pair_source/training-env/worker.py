"""Candidate training process. Never computes held-out reward or loads final data."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from checks import InvalidCandidate, import_candidate, check_model, check_logits, check_causal
from protocol import get_protocol
from tensor_hashes import tensor_state_sha256, tensor_state_spec_sha256, state_change_evidence


def classify_exception(exc, phase):
    """Map failures to stable protocol statuses; generic runtime failures stay candidate-invalid."""
    if phase in {'checkpoint-write', 'train-data', 'heldout-data'} and not isinstance(exc, InvalidCandidate):
        status = 'error'
    elif isinstance(exc, InvalidCandidate):
        status = 'invalid'
    elif isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        status = 'error'
    elif isinstance(exc, RuntimeError):
        message = str(exc).lower()
        backend_markers = ('mps device not available', 'cuda is not available',
                           'device lost', 'device was lost',
                           'backend is unavailable', 'backend unavailable', 'no kernel image')
        status = 'blocked' if any(marker in message for marker in backend_markers) else 'invalid'
    elif isinstance(exc, (SyntaxError, TypeError, AttributeError, NameError, ValueError, ImportError, ArithmeticError)):
        status = 'invalid'
    else:
        status = 'error'
    return {'status': status, 'phase': phase,
            'reason': f'{type(exc).__name__}: {exc}'}


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def batch(data, starts, context, device):
    ids = np.stack([data[s:s + context + 1] for s in starts]).astype(np.int64)
    tensor = torch.from_numpy(ids).to(device)
    return tensor[:, :-1], tensor[:, 1:]


def train(args, phase_state=None):
    phase_state = phase_state if phase_state is not None else {}
    phase_state['phase'] = 'setup'
    p = get_protocol(args.profile, getattr(args, "diagnostic", None))
    if getattr(p, "diagnostic", None) and args.device != "cpu":
        raise InvalidCandidate("cpu-training-v1 diagnostic requires device cpu")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    phase_state['phase'] = 'train-data'
    data = np.load(args.train, mmap_mode='r', allow_pickle=False)
    phase_state['phase'] = 'candidate-import'
    sol = import_candidate(args.candidate)
    phase_state['phase'] = 'candidate-build'
    model = sol.build_model().to(args.device)
    count = check_model(model)
    phase_state['phase'] = 'train-data'
    x, y = batch(data, rng.integers(0, len(data) - p.context, p.batch), p.context, args.device)
    phase_state['phase'] = 'candidate-forward'
    check_causal(model, x)
    phase_state['phase'] = 'candidate-build'
    # Opt-in league evidence binds seeded model state before optimizer code runs.
    state_evidence = getattr(args, 'state_evidence', False)
    initial_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                     if state_evidence else None)
    opt = sol.build_optimizer(model)
    if state_evidence and tensor_state_spec_sha256(initial_state) != tensor_state_spec_sha256(model.state_dict()):
        raise InvalidCandidate('optimizer changed frozen registered model state structure')
    if state_evidence and tensor_state_sha256(initial_state) != tensor_state_sha256(model.state_dict()):
        raise InvalidCandidate('optimizer changed frozen initial model weights')
    # Evaluator owns update count and batches. Candidate-returned loss is ignored.
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    if getattr(p, "diagnostic", None) or state_evidence:
        if initial_state is None:
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        phase_state['phase'] = 'checkpoint-write'
        initial_checkpoint = Path(args.checkpoint).with_name('initial-weights.pt')
        if initial_checkpoint == Path(args.checkpoint):
            raise InvalidCandidate('final checkpoint must differ from initial-weights.pt')
        torch.save(initial_state, initial_checkpoint)
    sync(args.device)
    started = time.monotonic()
    for step in range(p.steps):
        phase_state['phase'] = 'train-data'
        x, y = batch(data, rng.integers(0, len(data) - p.context, p.batch), p.context, args.device)
        phase_state['phase'] = 'candidate-training'
        sol.train_step(model, opt, x, y, step / p.steps)
        sync(args.device)
        if time.monotonic() - started > p.train_seconds_limit:
            raise InvalidCandidate('training exceeded protocol time limit')
        check_model(model)
        if time.monotonic() - started > p.train_seconds_limit:
            raise InvalidCandidate('training exceeded protocol time limit during validation')
    seconds = time.monotonic() - started
    changed = any(k not in initial or not torch.equal(v, initial[k]) for k, v in model.named_parameters())
    if not changed:
        raise InvalidCandidate('training did not change any registered parameters')
    phase_state['phase'] = 'candidate-forward'
    check_causal(model, x)
    check_logits(model(x), x)
    if time.monotonic() - started > p.train_seconds_limit:
        raise InvalidCandidate('training exceeded protocol time limit during final validation')
    validated_seconds = time.monotonic() - started
    if state_evidence and tensor_state_spec_sha256(initial_state) != tensor_state_spec_sha256(model.state_dict()):
        raise InvalidCandidate('training changed frozen registered model state structure')
    phase_state['phase'] = 'checkpoint-write'
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.checkpoint)
    evidence = {}
    if initial_state is not None:
        final_state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        evidence = {'initial_tensor_sha256': tensor_state_sha256(initial_state),
                    'final_tensor_sha256': tensor_state_sha256(final_state),
                    'initial_checkpoint': str(initial_checkpoint),
                    **state_change_evidence(initial_state, final_state)}
        if state_evidence:
            evidence['initial_state_capture'] = 'after-causal-check-before-optimizer-build'
            evidence['initial_state_spec_sha256'] = tensor_state_spec_sha256(initial_state)
            evidence['final_state_spec_sha256'] = tensor_state_spec_sha256(final_state)
    return {**evidence, 'status': 'ok', 'phase': 'train-complete', 'params': count, 'steps': p.steps, 'train_seconds': seconds,
            'training_tokens': p.steps * p.batch * p.context, 'parameters_changed': True,
            'train_with_final_validation_seconds': validated_seconds}


def main():
    ap = argparse.ArgumentParser()
    for name in ('candidate', 'train', 'checkpoint', 'result', 'profile', 'device'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--diagnostic')
    ap.add_argument('--state-evidence', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    phase_state = {'phase': 'setup'}
    try:
        result = train(args, phase_state)
    except InvalidCandidate as exc:
        result = {'status': 'invalid', 'reason': str(exc), 'phase': phase_state['phase']}
    except Exception as exc:
        result = classify_exception(exc, phase_state['phase'])
    Path(args.result).write_text(json.dumps(result, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
