"""Establish an idle baseline, then compare it before every standard run."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from suite.records import file_sha256
from suite.resources import accelerator_lock, known_competitors
from protocol import get_protocol, VERSION
from worker import sync
import starter


def reference(device, repeats=5):
    torch.set_num_threads(2)
    torch.manual_seed(9137)
    model = starter.build_model().to(device)
    opt = starter.build_optimizer(model)
    x = torch.randint(0, 4096, (32, 256), device=device)
    y = torch.roll(x, -1, 1)
    starter.train_step(model, opt, x, y, 0.1)
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.monotonic()
        starter.train_step(model, opt, x, y, 0.1)
        sync(device)
        times.append(time.monotonic() - start)
    return times


def identity(device):
    return {'version': VERSION, 'device': device, 'torch': torch.__version__,
            'hardware': platform.platform(), 'starter_sha256': file_sha256(HERE / 'starter.py')}


def verify_calibration(path, device):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('identity') != identity(device) or receipt.get('idle_assertion') is not True:
        raise RuntimeError('calibration identity mismatch or absent idle assertion; recalibrate')
    if time.time() - receipt['created_unix'] > 24 * 3600:
        raise RuntimeError('calibration older than 24 hours; recalibrate while idle')
    baseline = receipt.get('median_seconds')
    if not isinstance(baseline, (int, float)) or not 0 < baseline < float('inf'):
        raise RuntimeError('invalid calibration median')
    if known_competitors():
        raise RuntimeError('known competing workload during calibration check')
    times = reference(device, repeats=3)
    ratio = statistics.median(times) / baseline
    spread = (max(times) - min(times)) / statistics.median(times)
    if not 0.9 <= ratio <= 1.1 or spread > 0.1 or known_competitors():
        raise RuntimeError(f'calibration drift/noise: ratio={ratio:.3f}, spread={spread:.3f}; standard blocked')
    return {'receipt_sha256': file_sha256(path), 'reference_seconds': times,
            'median_ratio': ratio, 'relative_spread': spread, 'idle_receipt': receipt}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--idle-confirmed', action='store_true', help='operator confirms other heavy jobs are absent')
    args = ap.parse_args()
    target = Path(args.output)
    if target.exists():
        ap.error('output already exists; preserve old calibration')
    if not args.idle_confirmed:
        ap.error('calibrate only on an idle machine, then pass --idle-confirmed')
    with accelerator_lock(device=args.device, timeout_seconds=1):
        if known_competitors():
            ap.error('known competing workloads; do not calibrate under load')
        times = reference(args.device)
        median = statistics.median(times)
        if (max(times) - min(times)) / median > 0.1 or known_competitors():
            ap.error('reference noise exceeds 10% or a competing workload appeared')
        receipt = {'identity': identity(args.device), 'created_unix': time.time(),
                   'idle_assertion': True, 'reference_seconds': times, 'median_seconds': median}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt))

if __name__ == '__main__':
    main()
