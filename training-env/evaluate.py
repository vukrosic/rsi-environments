"""Independent evaluator entry point. Candidate Python is trusted cooperative code."""
import argparse
import contextlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from suite.records import make_record, write_record, file_sha256, loss_reward
from suite.resources import accelerator_lock, AcceleratorBusy, known_competitors
from data import load_data
from protocol import get_protocol


def child(script, arguments, output, timeout):
    """Terminate only this evaluator's child process group on a budget overrun."""
    with (output / f'{script}.stdout.log').open('w') as stdout, (output / f'{script}.stderr.log').open('w') as stderr:
        process = subprocess.Popen([sys.executable, str(HERE / script), *arguments], stdout=stdout,
                                   stderr=stderr, start_new_session=True, cwd=output)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {'status': 'invalid', 'reason': 'candidate process exceeded wall-time budget'}
    if code:
        return {'status': 'error', 'reason': f'{script} exited {code}; see stderr log'}
    result_path = Path(arguments[arguments.index('--result') + 1])
    if not result_path.exists() or result_path.stat().st_size > 1_000_000:
        return {'status': 'error', 'reason': 'missing or excessive subprocess result'}
    return json.loads(result_path.read_text())


def device_available(device):
    return device == 'cpu' or (device == 'mps' and torch.backends.mps.is_available()) or (device.startswith('cuda') and torch.cuda.is_available())


def evaluate(candidate, data_dir, output_dir, profile='smoke', split='dev', device='cpu', seed=0, calibration=None):
    p = get_protocol(profile)
    if split not in {'dev', 'final'}:
        raise ValueError('split must be dev or final')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)  # immutable receipts; never overwrite a previous run
    metrics = {'protocol': p.record(), 'split': split}
    provenance = {'device': device, 'seed': seed, 'python': sys.version.split()[0],
                  'torch': torch.__version__, 'numpy': np.__version__,
                  'hardware': platform.platform(), 'machine': platform.machine(),
                  'evaluator_sha256': {name: file_sha256(HERE / name) for name in
                      ('evaluate.py', 'worker.py', 'grader.py', 'protocol.py', 'checks.py', 'data.py')},
                  'suite_sha256': {name: file_sha256(HERE.parent / 'suite' / name) for name in ('records.py', 'resources.py')}}
    status, reward = 'error', None
    try:
        source = Path(candidate).resolve()
        snapshot = out / 'candidate.py'
        snapshot.write_bytes(source.read_bytes())
        provenance['source_sha256'] = file_sha256(snapshot)
        try:
            _, _, _, manifest = load_data(data_dir, profile)
        except ValueError as exc:
            raise RuntimeError(f'data assets invalid or missing: {exc}') from exc
        provenance['data_manifest'] = manifest
        provenance['data_manifest_sha256'] = file_sha256(Path(data_dir) / 'manifest.json')
        provenance['data_kind'] = manifest['kind']
        if not device_available(device):
            raise RuntimeError(f'device unavailable: {device}')
        with accelerator_lock(device=device, timeout_seconds=1) as lease:
            provenance['lease'] = lease
            before = known_competitors()
            provenance['contention_before'] = before
            if profile == 'standard' and before:
                raise RuntimeError('known competing workloads; standard run blocked')
            if profile == 'standard':
                if calibration is None:
                    raise RuntimeError('standard requires an idle-device calibration receipt; use calibrate.py')
                from calibrate import verify_calibration
                provenance['calibration'] = verify_calibration(calibration, device)
            common = ['--candidate', str(snapshot), '--profile', profile, '--device', device, '--seed', str(seed)]
            checkpoint = out / 'weights.pt'
            training = child('worker.py', [*common, '--train', str(Path(data_dir).resolve() / 'train.npy'),
                             '--checkpoint', str(checkpoint), '--result', str(out / 'train-result.json')], out, p.process_timeout)
            status = training['status']
            metrics.update(training)
            if status == 'ok':
                if checkpoint.stat().st_size > 100_000_000:
                    status = 'invalid'
                    metrics['reason'] = 'checkpoint exceeds 100 MB'
                else:
                    provenance['checkpoint_sha256'] = file_sha256(checkpoint)
                    grading = child('grader.py', [*common, '--checkpoint', str(checkpoint),
                        '--heldout', str(Path(data_dir).resolve() / f'{split}.npy'),
                        '--result', str(out / 'eval-result.json')], out, p.process_timeout)
                    status = grading['status']
                    metrics.update(grading)
            try:
                _, _, _, verified = load_data(data_dir, profile)
                if file_sha256(Path(data_dir) / 'manifest.json') != provenance['data_manifest_sha256']:
                    raise ValueError('manifest changed during execution')
                provenance['post_validation'] = verified.get('validation', {})
            except ValueError as exc:
                status = 'invalid'
                metrics['reason'] = f'data changed during candidate execution: {exc}'
            provenance['contention_after'] = known_competitors()
            if profile == 'standard' and provenance['contention_after']:
                status = 'blocked'
                metrics['reason'] = 'contention appeared during run; do not compare this reward'
            if file_sha256(snapshot) != provenance['source_sha256']:
                status = 'invalid'
                metrics['reason'] = 'frozen candidate source changed during execution'
            reward = loss_reward(metrics['held_out_loss']) if status == 'ok' else (0.0 if status == 'invalid' else None)
    except (FileNotFoundError, RuntimeError, AcceleratorBusy) as exc:
        status, reward = 'blocked', None
        metrics['reason'] = str(exc)
    except Exception as exc:
        status, reward = 'error', None
        metrics['reason'] = f'{type(exc).__name__}: {exc}'
    metrics.pop('status', None)
    record = make_record(environment='training', task='tinystories-next-token', profile=profile,
                         status=status, reward=reward, metrics=metrics, provenance=provenance)
    write_record(out / 'record.json', record)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', default=str(HERE / 'starter.py'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--profile', choices=('smoke', 'standard'), default='standard')
    ap.add_argument('--split', choices=('dev', 'final'), default='dev')
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--calibration')
    args = ap.parse_args()
    record = evaluate(args.candidate, args.data, args.output, args.profile, args.split, args.device, args.seed, args.calibration)
    print(json.dumps(record, allow_nan=False))
    return 0 if record['status'] == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
