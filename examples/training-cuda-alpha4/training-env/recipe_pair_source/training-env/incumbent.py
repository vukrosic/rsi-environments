"""The code every agent starts from: the model and one training step.
score.py trains it for exactly 500 steps and measures the loss on held-out text."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, CTX = 4096, 256  # fixed by the data
D, HEADS, LAYERS = 256, 4, 5
LR, WARMUP_FRAC, WD = 1e-3, 0.03, 0.1


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc, self.out = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, D), nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        """idx: (batch, time) token ids. Returns logits (batch, time, VOCAB)."""
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def build_model():
    return GPT()


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))


def train_step(model, opt, x, y, frac):
    """One update on one batch. x, y: (32, 256) token ids, y is x shifted by one.
    frac: share of training already done, from 0 to 1. Returns the loss."""
    lr = LR * frac / WARMUP_FRAC if frac < WARMUP_FRAC else \
        LR * 0.5 * (1 + math.cos(math.pi * (frac - WARMUP_FRAC) / (1 - WARMUP_FRAC)))
    for g in opt.param_groups:
        g["lr"] = lr
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss
import atexit as _audit_atexit
import json as _audit_json
import platform as _audit_platform
import resource as _audit_resource
import sys as _audit_sys
from pathlib import Path as _AuditPath

_audit_original_build_model = build_model
_audit_original_build_optimizer = build_optimizer
_audit_original_train_step = train_step
_audit_role = 'worker' if '--train' in _audit_sys.argv else ('grader' if '--heldout' in _audit_sys.argv else 'unknown')
_audit_path = _AuditPath.cwd() / (_audit_role + '-operator.json')
_audit_selected_train_calls = {1, 2, 250, 499, 500}
_audit_forward_limit = 12
_audit_grader_scored_forward_labels = {
    7: 'standard-heldout-scoring-first-batch',
    66: 'standard-heldout-scoring-last-batch',
}
_audit_receipt = {
    'schema': 'cuda-operator-audit-v1', 'role': _audit_role,
    'torch_version': str(torch.__version__),
    'torch_cuda_version': getattr(torch.version, 'cuda', None),
    'python_version': _audit_sys.version.split()[0],
    'platform': _audit_platform.platform(), 'threads': torch.get_num_threads(),
    'interop_threads': torch.get_num_interop_threads(),
    'cuda_inventory': [], 'cuda_inventory_errors': [],
    'forwards_total': 0, 'forwards': [], 'train_calls_total': 0,
    'source_bound_forward_plan': {
        'assumed_standard_grader_total': 66,
        'assumed_causal_check_forwards': 6,
        'assumed_heldout_scoring_forwards': 60,
        'phase_labels_are_source_bound_assumptions': True,
    },
    'train_calls': [], 'audit_errors': [],
    'progress': {'expected_train_calls': 500, 'last_train_call_started': 0,
                 'last_train_call_completed': 0, 'complete': False},
    'initial_state_evidence': {
        'source': 'native --state-evidence supplied by the A2000 session',
        'duplicate_checkpoint_written': False},
}
_audit_context_train_call = None
_audit_open_forwards = []


def _audit_meta(tensor):
    return {'device': str(tensor.device), 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}


def _audit_summary(tensors):
    counts = {}
    count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        key = str(tensor.device) + '/' + str(tensor.dtype)
        counts[key] = counts.get(key, 0) + 1
        count += 1
    return {'count': count, 'devices_dtypes': counts}


def _audit_cuda_inventory():
    try:
        _audit_receipt['cuda_inventory'] = {
            'available': bool(torch.cuda.is_available()),
            'device_count': int(torch.cuda.device_count()),
            'current_device': (int(torch.cuda.current_device()) if torch.cuda.is_available() else None),
            'devices': [],
            'cudnn_version': torch.backends.cudnn.version(),
            'cuda_driver_version_code': (int(torch._C._cuda_getDriverVersion())
                                         if hasattr(torch._C, '_cuda_getDriverVersion') else None),
            'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        }
        for index in range(_audit_receipt['cuda_inventory']['device_count']):
            props = torch.cuda.get_device_properties(index)
            _audit_receipt['cuda_inventory']['devices'].append({
                'index': index, 'name': props.name,
                'compute_capability': [int(props.major), int(props.minor)],
                'total_memory_bytes': int(props.total_memory),
                'multi_processor_count': int(props.multi_processor_count),
            })
    except Exception as exc:
        _audit_receipt['cuda_inventory_errors'].append(type(exc).__name__ + ': ' + str(exc)[:200])


def _audit_resources():
    result = {
        'peak_rss': int(_audit_resource.getrusage(_audit_resource.RUSAGE_SELF).ru_maxrss),
        'peak_rss_unit': 'bytes' if _audit_platform.system() == 'Darwin' else 'KiB',
    }
    try:
        result['cuda_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        result['cuda_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        result['cuda_current_allocated_bytes'] = int(torch.cuda.memory_allocated())
        result['cuda_current_reserved_bytes'] = int(torch.cuda.memory_reserved())
    except Exception as exc:
        result['cuda_memory_error'] = type(exc).__name__ + ': ' + str(exc)[:200]
    return result


def _audit_flush():
    try:
        _audit_receipt['resources'] = _audit_resources()
        temp = _audit_path.with_suffix('.json.tmp')
        temp.write_text(_audit_json.dumps(_audit_receipt, allow_nan=False, indent=2) + '\n')
        temp.replace(_audit_path)
    except Exception as exc:
        _audit_receipt['audit_errors'].append(type(exc).__name__ + ': ' + str(exc)[:200])


def _audit_attempt(action):
    try:
        action()
    except Exception as exc:
        _audit_receipt['audit_errors'].append(type(exc).__name__ + ': ' + str(exc)[:200])


_audit_cuda_inventory()
_audit_atexit.register(_audit_flush)


def build_model():
    model = _audit_original_build_model()

    def before(module, inputs):
        _audit_receipt['forwards_total'] += 1
        train_call = _audit_context_train_call
        # Keep a bounded set: the first two calls, plus examples belonging to
        # selected train calls. Hooks only retain metadata, never tensor values.
        forward_index = _audit_receipt['forwards_total']
        keep = (_audit_receipt['forwards_total'] <= 2 or
                (_audit_role == 'grader' and forward_index in _audit_grader_scored_forward_labels) or
                (train_call in _audit_selected_train_calls and
                 len(_audit_receipt['forwards']) < _audit_forward_limit))
        if keep and inputs and torch.is_tensor(inputs[0]):
            entry = {
                'call': _audit_receipt['forwards_total'],
                'train_call': train_call,
                'source_bound_assumed_phase': _audit_grader_scored_forward_labels.get(forward_index),
                'input': _audit_meta(inputs[0]),
                'parameters': _audit_summary(module.parameters()),
                'completed': False,
            }
            _audit_receipt['forwards'].append(entry)
            _audit_open_forwards.append(entry)

    def after(module, inputs, output):
        if _audit_open_forwards and torch.is_tensor(output):
            entry = _audit_open_forwards.pop()
            entry['logits'] = _audit_meta(output)
            entry['completed'] = True

    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    _audit_attempt(lambda: _audit_receipt.update({'build_parameters': _audit_summary(model.parameters())}))
    return model


def build_optimizer(model):
    optimizer = _audit_original_build_optimizer(model)
    _audit_attempt(lambda: _audit_receipt.update({
        'optimizer_build_parameters': _audit_summary(model.parameters()),
        'optimizer_type': type(optimizer).__name__,
        'optimizer_defaults': {
            key: value for key, value in optimizer.defaults.items()
            if key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'fused')
        },
    }))
    _audit_flush()
    return optimizer


def train_step(model, opt, x, y, frac):
    call = _audit_receipt['train_calls_total'] + 1
    _audit_receipt['train_calls_total'] = call
    _audit_receipt['progress']['last_train_call_started'] = call
    selected = call in _audit_selected_train_calls
    entry = None
    if selected:
        entry = {
            'call': call, 'x': _audit_meta(x), 'y': _audit_meta(y),
            'fraction': float(frac),
            'parameters_before': _audit_summary(model.parameters()),
            'completed': False,
        }
        _audit_receipt['train_calls'].append(entry)
    global _audit_context_train_call
    previous_context = _audit_context_train_call
    _audit_context_train_call = call
    try:
        result = _audit_original_train_step(model, opt, x, y, frac)
    except BaseException:
        _audit_context_train_call = previous_context
        if selected:
            _audit_flush()
        raise
    finally:
        _audit_context_train_call = previous_context

    def record():
        if entry is None:
            _audit_receipt['progress']['last_train_call_completed'] = call
            return
        entry['parameters_after'] = _audit_summary(model.parameters())
        entry['gradients'] = _audit_summary(parameter.grad for parameter in model.parameters())
        by_key = {}
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    by_key.setdefault(str(key), []).append(value)
        entry['optimizer_state'] = {key: _audit_summary(values) for key, values in by_key.items()}
        entry['optimizer_state_semantics'] = {
            'step': 'scalar counter; CPU placement is allowed',
            'exp_avg': 'Adam first moment; inspect actual tensor device',
            'exp_avg_sq': 'Adam second moment; inspect actual tensor device',
        }
        entry['resources'] = _audit_resources()
        entry['completed'] = True
        _audit_receipt['progress']['last_train_call_completed'] = call
        if call >= 500:
            _audit_receipt['progress']['complete'] = True

    _audit_attempt(record)
    if selected:
        _audit_flush()
    return result
