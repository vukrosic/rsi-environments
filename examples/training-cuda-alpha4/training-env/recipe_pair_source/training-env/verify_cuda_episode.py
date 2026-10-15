"""Verify saved standard CUDA evidence; no training, grading or CUDA tensors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    p = Path(path)
    if p.stat().st_size > 1_000_000:
        raise ValueError('excessive proof JSON')
    return json.loads(p.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite proof')))


def cuda_tensor(meta):
    if not isinstance(meta, dict) or not str(meta.get('device', '')).startswith('cuda:'):
        raise ValueError('actual CUDA tensor placement missing')


def cuda_summary(meta):
    counts = meta.get('devices_dtypes', {})
    if type(meta.get('count')) is not int or meta['count'] <= 0 or not counts:
        raise ValueError('empty CUDA parameter/gradient/moment proof')
    if sum(counts.values()) != meta['count'] or any(not key.startswith('cuda:') for key in counts):
        raise ValueError('mixed or non-CUDA parameter/gradient/moment proof')


def verify_proofs(record, worker, grader, packet):
    m, p = record['metrics'], record['provenance']
    if (record['status'], record['profile'], p['device'], m['split']) != ('ok', 'standard', 'cuda', 'dev'):
        raise ValueError('successful standard CUDA dev receipt required')
    if m['protocol'] != packet['budget'] or (m['steps'], m['training_tokens'], m['eval_tokens']) != (500, 4096000, 491520):
        raise ValueError('actual standard call/token budget mismatch')
    if p['source_sha256'] != packet['candidate_sha256'] or p['data_manifest_sha256'] != packet['data_manifest_sha256']:
        raise ValueError('source/data identity mismatch')
    if p['evaluator_sha256'] != packet['evaluator_sha256'] or p['suite_sha256'] != packet['suite_sha256']:
        raise ValueError('native/shared source identity mismatch')
    if p['torch'] != packet['target_runtime']['torch'] or p['numpy'] != packet['target_runtime']['numpy'] or p['python'] != packet['target_runtime']['python']:
        raise ValueError('new CUDA runtime identity mismatch')
    if p['data_kind'] != 'real' or not p.get('lease', {}).get('accelerator_lease'):
        raise ValueError('real data and native accelerator lease required')
    if p.get('contention_before') or p.get('contention_after'):
        raise ValueError('standard observed contention excluded')
    loss = m['held_out_loss']
    if type(loss) not in (float, int) or not math.isfinite(loss) or loss < 0 or not math.isclose(record['reward'], math.exp(-loss), rel_tol=1e-12):
        raise ValueError('independent loss/native reward mismatch')
    if not m.get('initial_tensor_sha256') or m['final_tensor_sha256'] != m['loaded_final_tensor_sha256']:
        raise ValueError('initial/final/grader tensor evidence missing')
    if m['initial_state_spec_sha256'] != m['final_state_spec_sha256'] or m['params'] != 5063424:
        raise ValueError('fixed model structure mismatch')
    for proof, role in ((worker, 'worker'), (grader, 'grader')):
        if proof['schema'] != 'cuda-operator-audit-v1' or proof['role'] != role or proof['audit_errors'] or proof['cuda_inventory_errors']:
            raise ValueError('incomplete operator audit')
        if proof['torch_version'] != p['torch'] or proof['python_version'] != p['python']:
            raise ValueError('operator/native runtime mismatch')
        inv = proof['cuda_inventory']
        if inv['available'] is not True or not inv['devices'] or 'A2000' not in inv['devices'][0]['name']:
            raise ValueError('actual selected A2000 inventory missing')
        res = proof['resources']
        if res['peak_rss_unit'] != 'KiB' or res['cuda_peak_allocated_bytes'] <= 0 or res['cuda_peak_reserved_bytes'] < res['cuda_peak_allocated_bytes']:
            raise ValueError('Linux RSS/CUDA allocator proof missing')
        for f in proof['forwards']:
            if f['completed'] is not True:
                raise ValueError('incomplete forward evidence')
            cuda_tensor(f['input']); cuda_tensor(f['logits']); cuda_summary(f['parameters'])
    if worker['train_calls_total'] != 500 or worker['progress']['last_train_call_completed'] != 500 or worker['progress']['complete'] is not True:
        raise ValueError('500 completed training calls not proven')
    selected = {row['call']: row for row in worker['train_calls']}
    if set(selected) != {1, 2, 250, 499, 500}:
        raise ValueError('selected standard training evidence missing')
    for row in selected.values():
        if row['completed'] is not True:
            raise ValueError('incomplete selected training call')
        cuda_tensor(row['x']); cuda_tensor(row['y'])
        for key in ('parameters_before', 'parameters_after', 'gradients'):
            cuda_summary(row[key])
        for key in ('exp_avg', 'exp_avg_sq'):
            cuda_summary(row['optimizer_state'][key])
        # Adam's scalar step may reside on CPU; its placement is recorded only.
    if grader['forwards_total'] != 66:
        raise ValueError('6 causal plus 60 standard score forwards not observed')
    scored = {row['call']: row for row in grader['forwards']}
    if not {7, 66}.issubset(scored):
        raise ValueError('first/last scored CUDA batch evidence missing')
    for index in (7, 66):
        if scored[index]['input']['shape'] != [32, 256] or scored[index]['logits']['shape'] != [32, 256, 4096]:
            raise ValueError('standard score tensor shape mismatch')
    return {'actual_tensor_metadata_consistent': True,
            'full_standard_session_accepted': False,
            'missing_session_checks': [
                'fresh calibration bound to source, runtime, physical GPU, and driver',
                'outer supervisor record',
                'owned-job cleanup evidence',
            ],
            'held_out_loss': loss, 'native_reward': record['reward'],
            'memory_scope': 'Per-process PyTorch CUDA allocator peaks; RSS is separate host memory; no speed comparison'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--packet', required=True)
    ap.add_argument('--training-root', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        ap.error('preserve existing verification; select a fresh output')
    run, root = Path(args.run), Path(args.training_root)
    packet, record = read(args.packet), read(run / 'record.json')
    worker, grader = read(run / 'worker-operator.json'), read(run / 'grader-operator.json')
    result = verify_proofs(record, worker, grader, packet)
    if sha(run / 'candidate.py') != packet['candidate_sha256']:
        raise ValueError('candidate snapshot file mismatch')
    sys.path.insert(0, str(root))
    from tensor_hashes import tensor_state_sha256, tensor_state_spec_sha256
    import torch
    weights = {}
    for file, prefix, prov in [('initial-weights.pt', 'initial', 'initial_checkpoint_sha256'), ('weights.pt', 'final', 'checkpoint_sha256')]:
        path = run / file
        if sha(path) != record['provenance'][prov]:
            raise ValueError('checkpoint file mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if len(state) != 65 or not all(torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('saved tensor specification/finiteness mismatch')
        tensor, spec = tensor_state_sha256(state), tensor_state_spec_sha256(state)
        if tensor != record['metrics'][prefix + '_tensor_sha256'] or spec != record['metrics'][prefix + '_state_spec_sha256']:
            raise ValueError('saved canonical tensor/spec mismatch')
        weights[prefix] = {'file_sha256': sha(path), 'tensor_sha256': tensor, 'state_spec_sha256': spec}
    if weights['initial']['tensor_sha256'] == weights['final']['tensor_sha256']:
        raise ValueError('no inner-model weight change')
    result.update(saved_checkpoint_consistency=True, weights=weights,
                  source_packet_sha256=sha(args.packet), record_sha256=sha(run / 'record.json'))
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
