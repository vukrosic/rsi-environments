"""Separate evaluator: recompute next-token CE from frozen trained weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import cross_entropy as evaluator_cross_entropy
from checks import InvalidCandidate, import_candidate, check_model, check_logits, check_causal
from protocol import get_protocol, VOCAB
from tensor_hashes import tensor_state_sha256
from worker import batch, classify_exception


def grade(args, phase_state=None):
    phase_state = phase_state if phase_state is not None else {}
    phase_state['phase'] = 'setup'
    p = get_protocol(args.profile, getattr(args, "diagnostic", None))
    if getattr(p, "diagnostic", None) and args.device != "cpu":
        raise InvalidCandidate("cpu-training-v1 diagnostic requires device cpu")
    reference = getattr(args, "reference_checkpoint", None)
    if reference and not getattr(p, "diagnostic", None):
        raise InvalidCandidate("reference checkpoint comparison is diagnostic-only")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    phase_state['phase'] = 'candidate-import'
    candidate = import_candidate(args.candidate)
    phase_state['phase'] = 'candidate-build'
    model = candidate.build_model().to(args.device)
    check_model(model)
    phase_state['phase'] = 'checkpoint-read'
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    count = check_model(model)
    loaded_hash = tensor_state_sha256(model.state_dict())
    phase_state['phase'] = 'heldout-data'
    data = np.load(args.heldout, mmap_mode='r', allow_pickle=False)
    x, _ = batch(data, [i * (p.context + 1) for i in range(p.batch)], p.context, args.device)
    phase_state['phase'] = 'candidate-forward'
    check_causal(model, x)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(p.eval_batches):
            phase_state['phase'] = 'heldout-data'
            starts = [(i * p.batch + j) * (p.context + 1) for j in range(p.batch)]
            x, y = batch(data, starts, p.context, args.device)
            phase_state['phase'] = 'candidate-forward'
            logits = check_logits(model(x), x)
            loss = evaluator_cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1)).item()
            if not np.isfinite(loss) or loss < 0:
                raise InvalidCandidate('nonfinite held-out loss')
            total += loss
    evidence = {'loaded_final_tensor_sha256': loaded_hash}
    if reference:
        reference_args = argparse.Namespace(**vars(args))
        reference_args.checkpoint = reference
        reference_args.reference_checkpoint = None
        before = grade(reference_args, phase_state)
        evidence.update({'diagnostic_comparison': True,
                         'reference_tensor_sha256': before['loaded_final_tensor_sha256'],
                         'before_held_out_loss': before['held_out_loss'],
                         'after_held_out_loss': total / p.eval_batches,
                         'held_out_loss_improvement': before['held_out_loss'] - total / p.eval_batches})
    return {**evidence, 'status': 'ok', 'phase': 'grade-complete', 'held_out_loss': total / p.eval_batches, 'params': count,
            'eval_tokens': p.eval_batches * p.batch * p.context}


def main():
    ap = argparse.ArgumentParser()
    for name in ('candidate', 'checkpoint', 'heldout', 'profile', 'device', 'result'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--diagnostic')
    ap.add_argument('--reference-checkpoint')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    phase_state = {'phase': 'setup'}
    try:
        result = grade(args, phase_state)
    except InvalidCandidate as exc:
        result = {'status': 'invalid', 'phase': phase_state['phase'], 'reason': str(exc)}
    except Exception as exc:
        result = classify_exception(exc, phase_state['phase'])
    Path(args.result).write_text(json.dumps(result, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
