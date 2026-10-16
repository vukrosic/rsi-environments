"""Canonical tensor-state hashes independent of checkpoint serialization."""
import hashlib
import math
import json
import struct
import torch


def tensor_state_spec_sha256(state):
    """Bind registered state names, dtypes and shapes without copying payloads."""
    spec = [[key, str(state[key].dtype), list(state[key].shape)] for key in sorted(state)]
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)],
                            ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack('>Q', len(header)))
        digest.update(header)
        digest.update(struct.pack('>Q', len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def state_change_evidence(initial, final):
    """Scaled norms avoid squaring large/small finite float64 differences."""
    changed = [key for key in initial.keys() | final.keys()
               if key not in initial or key not in final or
               not torch.equal(initial[key], final[key])]
    norm = 0.0
    for key in initial.keys() & final.keys():
        if initial[key].shape != final[key].shape:
            continue
        dtype = torch.complex128 if initial[key].is_complex() or final[key].is_complex() else torch.float64
        before = initial[key].detach().cpu().to(dtype)
        after = final[key].detach().cpu().to(dtype)
        difference = after - before
        if not torch.isfinite(difference).all():
            # Finite operands can have an unrepresentable difference/norm.
            return {'changed_state_tensors': len(changed), 'state_update_l2': None,
                    'state_update_l2_reason': 'norm exceeds finite float64 range'}
        scale = difference.abs().max().item() if difference.numel() else 0.0
        if scale:
            unit_norm = ((difference / scale).abs().square().sum().item()) ** 0.5
            norm = math.hypot(norm, scale * unit_norm)
            if not math.isfinite(norm):
                return {'changed_state_tensors': len(changed), 'state_update_l2': None,
                        'state_update_l2_reason': 'norm exceeds finite float64 range'}
    return {'changed_state_tensors': len(changed), 'state_update_l2': norm}
