"""Package an explicit source/receipt whitelist; never bundle dataset or agent logs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        ap.error('preserve existing release candidate; select a new output')
    files = [*HERE.glob('*.py'), *HERE.glob('*.md'), HERE / '.gitignore',
             HERE / 'requirements.txt', HERE / 'requirements-tested.txt',
             *HERE.glob('tests/*.py'), *HERE.glob('examples/**/*.py'),
             *HERE.glob('examples/**/record.json'), *HERE.glob('examples/**/summary.json')]
    # Explicit public verification checkpoints; never include worker logs/data/checkpoints.
    checkpoint = HERE / 'checkpoints' / '2026-10-14-semantic-v1'
    files.extend([*checkpoint.glob('*.md'), *checkpoint.glob('*.json'),
                  *checkpoint.glob('snapshots/**/*.py'),
                  checkpoint / 'backend-smoke' / 'receipt.json',
                  checkpoint / 'real-dev-smoke' / 'record.json',
                  checkpoint / 'real-dev-smoke' / 'candidate.py'])
    # Public continuous-improvement receipts only; exclude venvs, generated traces and logs.
    for name in ('2026-10-14-validation-cache', '2026-10-14-continuous'):
        public_checkpoint = HERE / 'checkpoints' / name
        files.extend(public_checkpoint.glob('*.md'))
        files.extend(public_checkpoint.glob('*.json'))
    for name in ('framework-frozen-train', 'framework-frozen-heldout', 'framework-frozen-null'):
        files.append(HERE / 'checkpoints/2026-10-14-continuous' / name / 'receipt.json')
    files.append(HERE / 'checkpoints/2026-10-14-continuous/repro/requirements-resolved.txt')
    suite = HERE.parent / 'suite'
    files.extend(suite / name for name in ('__init__.py', 'records.py', 'resources.py'))
    files = sorted(set(path for path in files if path.is_file()))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=Path('rsi') / path.relative_to(HERE.parent), recursive=False)
    receipt = {'artifact': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'files': [str(path.relative_to(HERE.parent)) for path in files],
               'dataset_bundled': False, 'agent_logs_bundled': False, 'published': False}
    target.with_suffix(target.suffix + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'artifact': str(target), 'sha256': receipt['sha256'], 'file_count': len(files)}))

if __name__ == '__main__':
    main()
