"""REMOTE ONLY: freeze host-dependent round fields before any calibration/model run.

Standard library only. Does not import candidates, Torch or NumPy. Asset hashing
runs on the GPU host, against its existing cache; no asset transfer is needed.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    packet = json.loads(Path(args.packet).read_text())
    if platform.system() != 'Linux' or sys.version.split()[0] != packet['target_runtime']['python']:
        raise ValueError('remote Linux interpreter does not match frozen runtime')
    for name in ('torch', 'numpy', 'tokenizers'):
        if metadata.version(name) != packet['target_runtime'][name]:
            raise ValueError('existing remote dependency mismatch: ' + name)
    for name, expected in packet['source_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('frozen source mismatch: ' + name)
        if name.endswith('.py'):
            compile((root / name).read_text(), name, 'exec')
    data = Path(args.data).resolve()
    if str(data) != packet['data_reuse']:
        raise ValueError('use the frozen existing remote TinyStories cache')
    for name, expected in packet['assets'].items():
        asset = data / name
        if asset.stat().st_size != expected['bytes'] or sha(asset) != expected['sha256']:
            raise ValueError('existing remote asset mismatch: ' + name)
    sys.path.insert(0, str(root / 'training-env'))
    from league_adapter import NATIVE, model_identity, runtime_identity, validate_round
    from protocol import get_protocol
    inc, challenger = (root / 'training-env' / name for name in ('incumbent.py', 'challenger.py'))
    needle = 'LR, WARMUP_FRAC, WD = 1e-3, 0.03, 0.1'
    if inc.read_text().count(needle) != 1 or challenger.read_text() != inc.read_text().replace(
            needle, 'LR, WARMUP_FRAC, WD = 2e-3, 0.03, 0.1'):
        raise ValueError('candidate delta differs from preregistered LR-only change')
    if not inc.read_text().startswith((root / 'training-env/starter.py').read_text()):
        raise ValueError('incumbent does not preserve exact starter prefix')
    identity = model_identity(inc)
    if model_identity(challenger) != identity:
        raise ValueError('frozen model/audit definitions differ')
    if get_protocol('standard').record() != packet['budget']:
        raise ValueError('standard budget changed')
    r = dict(schema_version=1, environment='training', round_id='cuda-recipe-lr2x-20261015-v1',
             workload_id='tinystories-next-token', incumbent_source=str(inc),
             incumbent_sha256=sha(inc), data_dir=str(data),
             data_manifest_sha256=packet['data_manifest_sha256'], profile='standard', device='cuda',
             seeds=packet['seeds'], validation_seeds=packet['validation_seeds'],
             validation_offset_tokens=packet['validation_offset_tokens'], budget=packet['budget'],
             runtime_sha256={name: sha(root / 'training-env' / name) for name in NATIVE},
             suite_sha256={name: sha(root / 'suite' / name) for name in ('records.py', 'resources.py')},
             min_improvement=packet['recipe']['min_improvement_nats_per_token'],
             comparison_timeout_seconds=packet['limits']['comparison_seconds'],
             calibration=str(Path(args.calibration).resolve()), model_identity=identity,
             oracle_sha256=sha(root / 'training-env/grader.py'),
             objective_name='development_cross_entropy', lower_is_better=True,
             runtime=runtime_identity('cuda'))
    validate_round(r)
    output = Path(args.output)
    save(output, r)
    for side, source in [('incumbent', inc), ('challenger', challenger)]:
        proof_packet = dict(budget=packet['budget'], candidate_sha256=sha(source),
                            data_manifest_sha256=packet['data_manifest_sha256'],
                            evaluator_sha256={name: r['runtime_sha256'][name] for name in NATIVE[:7]},
                            suite_sha256=r['suite_sha256'], target_runtime=packet['target_runtime'])
        save(output.parent / ('audit-' + side + '.json'), proof_packet)
    save(output.parent / 'preflight.json', dict(status='ok', packet_sha256=sha(args.packet),
         round_sha256=sha(output), source_count=len(packet['source_sha256']), asset_count=5,
         lr_only_delta=True, model_identity=identity, runtime=r['runtime'],
         actual_cuda_execution=False, calibration_verified=False, comparison_verified=False))
    print(json.dumps({'status': 'ok', 'round_sha256': sha(output), 'models_run': 0}))


if __name__ == '__main__':
    main()
