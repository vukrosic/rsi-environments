"""Root-relative source verification and optional metadata imports only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-imports', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'EXPORT-MANIFEST.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['target']
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('export member escaped source root')
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('exported source changed: ' + item['target'])
    result = dict(source_files_verified=len(manifest['files']), metadata_imports=False,
                  model_loaded=False, native_executed=False, installed_package=False)
    if args.metadata_imports:
        sys.path[:0] = [str(ROOT / 'consumer'), str(ROOT / 'training-env/group8-v1'),
                        str(ROOT / 'training-env'), str(ROOT)]
        import journal, transaction, native_bridge, learner, group_state, group_adapter
        expected = dict(journal=ROOT / 'consumer/journal.py', transaction=ROOT / 'consumer/transaction.py',
            native_bridge=ROOT / 'consumer/native_bridge.py', learner=ROOT / 'consumer/learner.py',
            group_state=ROOT / 'training-env/group8-v1/group_state.py',
            group_adapter=ROOT / 'training-env/group8-v1/group_adapter.py')
        for name, path in expected.items():
            if Path(sys.modules[name].__file__).resolve() != path:
                raise ValueError('mixed module origin: ' + name)
        if any(name in sys.modules for name in ('torch', 'numpy', 'transformers')):
            raise ValueError('metadata probe unexpectedly imported model dependencies')
        if group_adapter.CONTRACT != 'training-group8-public-v1':
            raise ValueError('public successor contract differs')
        result['metadata_imports'] = True
    print(json.dumps(result))


if __name__ == '__main__':
    main()
