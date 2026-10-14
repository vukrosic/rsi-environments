"""Fetch pinned model assets and record their byte hashes. No remote code."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def prepare():
    spec = json.loads((HERE / 'model.json').read_text())
    root = Path(snapshot_download(spec['id'], revision=spec['revision'],
        allow_patterns=['*.json', '*.txt', '*.safetensors', 'README.md']))
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(root.iterdir()) if p.is_file()}
    manifest = dict(spec, files=files)
    (HERE / 'assets-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return root

def local_model():
    spec = json.loads((HERE / 'model.json').read_text())
    root = Path(snapshot_download(spec['id'], revision=spec['revision'], local_files_only=True))
    manifest = json.loads((HERE / 'assets-manifest.json').read_text())
    assert manifest['revision'] == spec['revision'] and manifest['id'] == spec['id']
    for name, record in manifest['files'].items():
        if sha(root / name) != record['sha256']:
            raise ValueError(f'asset changed: {name}')
    return root

if __name__ == '__main__':
    print(prepare())
