"""Use live suite helpers in this checkout, bundled MIT copies standalone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from suite.resources import accelerator_lock
    from suite.records import make_record, speed_reward
except ModuleNotFoundError as exc:
    if exc.name not in ('suite', 'suite.resources', 'suite.records'):
        raise
    from shared.resources import accelerator_lock
    from shared.records import make_record, speed_reward

def normalize(raw):
    if raw['status'] in ('incorrect', 'diagnostic_incorrect'):
        status, reward = 'invalid', 0.0
    elif raw['status'] == 'failed':
        status, reward = 'error', None
    elif not raw['timing_valid']:
        status, reward = 'blocked', None
    else:
        status, reward = 'ok', speed_reward(raw['metrics']['geomean_speedup'])
    metrics = dict(raw.get('metrics', {}), correctness_pass=raw['ok'],
                   timing_valid=raw['timing_valid'], mode=raw['mode'],
                   evaluation_wall_s=raw['evaluation_wall_s'])
    metrics['contention_detected'] = any(m.get('blockers') for m in
        [raw.get('machine_before', {}), raw.get('machine_after', {})] + raw.get('conditions', []) +
        [m for w in raw.get('workers', {}).values() for m in w.get('conditions', [])])
    if 'error' in raw:
        metrics['error'] = raw['error']
    return make_record(environment='inference', task=raw['task'], profile='standard',
        status=status, reward=reward, metrics=metrics,
        provenance={'source_sha256': raw['source_sha256'], 'device': raw['device'],
                    'nonce': raw['nonce'], 'model': raw['model'],
                    'partition': raw.get('partition'), 'partition_task_id': raw.get('partition_task_id'),
                    'artifact_dir': raw['artifact_dir'], 'evidence_kind': raw['evidence_kind'],
                    'evaluator_hashes': raw['evaluator_hashes']})
