"""Source/runtime admission or one native terminal reward control; stdlib only."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {'torch': '2.11.0+cu128', 'numpy': '2.5.3', 'tokenizers': '0.22.1'}


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1000000:
        raise ValueError('bounded regular JSON required')
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
        raise ValueError('bounded regular source required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check():
    manifest = read(ROOT / 'SOURCE-MANIFEST.json')
    files = manifest['files']
    if not isinstance(files, dict) or not 1 <= len(files) <= 64:
        raise ValueError('source list bound')
    for name, pin in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or relative.as_posix() != name or any(p in {'.', '..', 'data', 'evidence', 'checkpoints', 'agent-chat'} or p.startswith('.') for p in relative.parts):
            raise ValueError('unsafe source name')
        path = ROOT / name
        if any(p.is_symlink() for p in path.parents if p != ROOT and ROOT in p.parents):
            raise ValueError('symlinked source ancestor')
        if path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise ValueError('source identity mismatch')
    if sys.version_info[:3] != (3, 12, 14):
        raise ValueError('existing Python3.12.14 required')
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError('existing runtime versions differ')
    return {'status': 'source_metadata_checked', 'source_files': len(files),
            'runtime_versions': actual, 'runtime_imported': False}


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def reward_check(path, data_manifest, expected_data_sha256=None):
    transition = read(path)
    data_hash = digest(data_manifest)
    if expected_data_sha256 is not None and data_hash != expected_data_sha256:
        raise ValueError('executor-bound public data manifest changed')
    info = transition['info']
    record = transition['record']
    metrics, provenance = record['metrics'], record['provenance']
    if transition.get('terminated') is not True or transition.get('truncated') is not False or info.get('trainable') is not True or info.get('exclude_from_rl') is not False:
        raise ValueError('terminal train eligibility absent')
    if info.get('task_split') != 'train' or info.get('scored_data_split') != 'dev':
        raise ValueError('unexpected task/data split')
    if any(record.get(k) != v for k, v in {'environment': 'training', 'profile': 'standard', 'status': 'ok', 'task': 'tinystories-next-token'}.items()):
        raise ValueError('native record not successful standard training')
    if provenance.get('device') != 'cuda' or provenance.get('torch') != EXPECTED['torch'] or provenance.get('numpy') != EXPECTED['numpy']:
        raise ValueError('native runtime/device mismatch')
    if provenance.get('data_kind') != 'real' or provenance.get('data_manifest_sha256') != data_hash or provenance.get('seed') != 14:
        raise ValueError('documented fresh-data/seed identity differs')
    if provenance.get('source_sha256') != digest(ROOT / 'training-env/starter.py'):
        raise ValueError('documented starter action identity differs')
    for name in ('evaluate.py', 'worker.py', 'grader.py', 'protocol.py', 'checks.py', 'data.py', 'tensor_hashes.py'):
        if provenance.get('evaluator_sha256', {}).get(name) != digest(ROOT / 'training-env' / name):
            raise ValueError('native evaluator source mismatch')
    for name in ('records.py', 'resources.py'):
        if provenance.get('suite_sha256', {}).get(name) != digest(ROOT / 'suite' / name):
            raise ValueError('native shared-helper source mismatch')
    if metrics.get('split') != 'dev' or metrics.get('protocol', {}).get('steps') != 500:
        raise ValueError('native fixed protocol mismatch')
    for key, expected in {'steps': 500, 'training_tokens': 4096000, 'eval_tokens': 491520, 'params': 5063424}.items():
        if type(metrics.get(key)) is not int or metrics[key] != expected:
            raise ValueError('native call/token/model count mismatch')
    loss, reward = metrics.get('held_out_loss'), record.get('reward')
    if not finite(loss) or loss < 0 or not finite(reward) or not finite(transition.get('reward')):
        raise ValueError('nonfinite loss/reward')
    if not math.isclose(reward, math.exp(-loss), rel_tol=1e-12, abs_tol=1e-15) or transition['reward'] != reward:
        raise ValueError('terminal/native reward formula mismatch')
    return {'status': 'native_terminal_reward_checked', 'reward': reward, 'development_loss': loss,
            'data_manifest_sha256': data_hash, 'steps': 500, 'training_tokens': 4096000, 'grading_tokens': 491520,
            'parameters': 5063424, 'trainable': True, 'outer_llm_updated': False,
            'learning_gain_measured': False, 'independent_session_acceptance': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transition', type=Path)
    ap.add_argument('--data-manifest', type=Path)
    ap.add_argument('--expected-data-sha256')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    try:
        result = source_check()
        if args.transition is not None:
            if args.data_manifest is None:
                raise ValueError('data manifest required for reward verification')
            result = reward_check(args.transition, args.data_manifest, args.expected_data_sha256)
    except (ValueError, OSError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
        result = {'status': 'blocked', 'reason_type': type(exc).__name__, 'reward': None, 'trainable': False}
    result.update(schema_version=1, release_verified=False)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write('\n')
    print(json.dumps({'status': result['status']}))
    return 2 if result['status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
