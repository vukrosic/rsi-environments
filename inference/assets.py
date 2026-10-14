"""Fetch pinned model assets and record their byte hashes. No remote code."""
import hashlib
import json
import re
from pathlib import Path
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
REQUIRED_FILES={'model.safetensors','config.json','tokenizer.json','tokenizer_config.json'}

def validate_manifest(manifest,spec):
    if not isinstance(manifest,dict) or any(manifest.get(key)!=spec[key] for key in ('id','revision','dtype','attention')):
        raise ValueError('model manifest does not match pinned model specification')
    files=manifest.get('files')
    if not isinstance(files,dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError('model manifest missing required weights/config/tokenizer files')
    for name,record in files.items():
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
            raise ValueError('manifest asset name must be a single filename')
        if not isinstance(record,dict) or type(record.get('bytes')) is not int or record['bytes']<=0:
            raise ValueError(f'invalid asset byte count: {name}')
        if not isinstance(record.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',record['sha256']):
            raise ValueError(f'invalid asset SHA256: {name}')
    return manifest

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
    validate_manifest(manifest,spec)
    (HERE / 'assets-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return root

def local_model():
    spec = json.loads((HERE / 'model.json').read_text())
    root = Path(snapshot_download(spec['id'], revision=spec['revision'], local_files_only=True))
    manifest = json.loads((HERE / 'assets-manifest.json').read_text())
    validate_manifest(manifest,spec)
    for name, record in manifest['files'].items():
        if (root/name).stat().st_size!=record['bytes']:
            raise ValueError(f'asset byte count changed: {name}')
        if sha(root / name) != record['sha256']:
            raise ValueError(f'asset changed: {name}')
    return root

if __name__ == '__main__':
    print(prepare())
