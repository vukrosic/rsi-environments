"""REMOTE ONLY companion controls around the unchanged native league adapter.

The coordinator supervises this process. No outer accelerator lease is held
across native calls: each native evaluator owns its usual common lease.
Saved tensor audits and independent complete-session acceptance are separate.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training-env'))
sys.path.insert(0, str(ROOT))
import league_adapter as league
from suite.resources import accelerator_lock, lock_path
from prepare_round import sha, save


def read(path):
    p = Path(path)
    if p.stat().st_size > 1_000_000:
        raise ValueError('excessive control JSON')
    return json.loads(p.read_text())


def query(fields, kind):
    result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, timeout=5, check=True)
    if len(result.stdout) > 64_000 or len(result.stderr) > 64_000:
        raise RuntimeError('excessive inventory output')
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def capture(label, packet, round_path, controls):
    path = controls / (label + '.json')
    result = {'label': label, 'created_unix': time.time(), 'status': 'blocked'}
    try:
        if str(lock_path()) != packet['lock_path']:
            raise RuntimeError('canonical common lease path required')
        with accelerator_lock(device='cuda', timeout_seconds=1) as lease:
            gpu = query('index,uuid,pci.bus_id,name,memory.total,driver_version,utilization.gpu', 'gpu')
            apps = query('pid,gpu_uuid,process_name,used_gpu_memory', 'compute-apps')
            if len(gpu) != 1 or len(gpu[0]) != 7 or 'A2000' not in gpu[0][3]:
                raise RuntimeError('selected single A2000 inventory required')
            result.update(gpu_identity=gpu[0][:6], sampled_gpu_utilization=gpu[0][6],
                          compute_apps=apps, lease=lease,
                          runtime=league.runtime_identity('cuda'),
                          source_sha256={name: sha(ROOT / name) for name in packet['source_sha256']},
                          asset_sha256={name: sha(Path(packet['data_reuse']) / name) for name in packet['assets']},
                          round_sha256=sha(round_path))
            if apps:
                raise RuntimeError('unrelated compute apps observed at native phase boundary')
            if result['source_sha256'] != packet['source_sha256']:
                raise RuntimeError('source closure drift')
            if result['asset_sha256'] != {name: row['sha256'] for name, row in packet['assets'].items()}:
                raise RuntimeError('real asset closure drift')
            initial_path = controls / 'before-calibration.json'
            if label != 'before-calibration' and not initial_path.exists():
                raise RuntimeError('before-calibration physical binding required')
            if initial_path.exists():
                initial = read(initial_path)
                for field in ('gpu_identity', 'runtime', 'source_sha256', 'asset_sha256', 'round_sha256'):
                    if initial.get('status') != 'ok' or result[field] != initial[field]:
                        raise RuntimeError('physical/source/asset/runtime binding drift: ' + field)
            calibration = Path(read(round_path)['calibration'])
            result['calibration_sha256'] = sha(calibration) if calibration.exists() else None
            calibrated_path = controls / 'after-calibration.json'
            if calibrated_path.exists():
                expected = read(calibrated_path)['calibration_sha256']
                if not expected or result['calibration_sha256'] != expected:
                    raise RuntimeError('calibration receipt changed or missing')
            if label == 'after-calibration' and not result['calibration_sha256']:
                raise RuntimeError('fresh calibration receipt missing')
        result['status'] = 'ok'
    except Exception as exc:
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
        save(path, result)
        raise
    save(path, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet', required=True)
    ap.add_argument('--round', required=True)
    ap.add_argument('--controls', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--capture', choices=('before-calibration', 'after-calibration', 'cleanup'))
    args = ap.parse_args()
    packet = read(args.packet)
    controls = Path(args.controls)
    controls.mkdir(parents=True, exist_ok=True)
    if args.capture:
        capture(args.capture, packet, args.round, controls)
        return
    calibrated = read(controls / 'after-calibration.json')
    if calibrated['status'] != 'ok':
        raise RuntimeError('successful fresh calibrated phase required')
    original = league.run_native
    native_controls = []

    def observed(source, r, out, seed, deadline, offset=0):
        before = capture('before-' + out.name, packet, args.round, controls)
        # The unchanged native supervisor owns the entire evaluator tree.
        try:
            value = original(source, r, out, seed, deadline, offset)
        finally:
            after = capture('after-' + out.name, packet, args.round, controls)
        native_controls.append({'run': out.name, 'before': before['status'], 'after': after['status'],
                                'raw_native_status': value['status'], 'raw_native_reward': value.get('reward')})
        if value['status'] != 'ok':
            raise RuntimeError('raw native non-ok retained; entire session excluded pending failure attribution')
        calibration = value['provenance'].get('calibration', {})
        ratio, spread = calibration.get('median_ratio'), calibration.get('relative_spread')
        if (calibration.get('receipt_sha256') != calibrated['calibration_sha256'] or
                type(ratio) not in (int, float) or not 0.9 <= ratio <= 1.1 or
                type(spread) not in (int, float) or not 0 <= spread <= 0.1):
            raise RuntimeError('native fresh calibration binding/gates missing')
        if value['provenance'].get('lease', {}).get('lock_path') != packet['lock_path']:
            raise RuntimeError('native lease differs from common physical session lease')
        return value

    league.run_native = observed
    result = league.compare(args.round, str(ROOT / 'training-env/challenger.py'), args.output)
    save(controls / 'pair-control-summary.json', {
        'status': result['status'], 'native_controls': native_controls,
        'comparison_sha256': sha(Path(args.output) / 'comparison.json'),
        'packet_sha256': sha(args.packet), 'round_sha256': sha(args.round),
        'provisional_comparison_eligible': result['comparison_eligible'],
        'effective_reward': None, 'reward_consumed': False, 'outer_policy_updated': False,
        'full_session_accepted': False,
        'remaining_acceptance': ['four saved-checkpoint/device audits on GPU host',
                                 'outer phase/job supervisors and final owned cleanup',
                                 'independent source-bound full-session review'],
        'failure_policy': 'Raw negatives preserved; missing controls or any infrastructure failure means null effective reward.',
        'load_limit': 'Boundary inventories and cooperative leases do not prove uninterrupted GPU idleness; strict native reference gates remain required.',
    })
    print(json.dumps({'status': result['status'], 'full_session_accepted': False}))


if __name__ == '__main__':
    main()
