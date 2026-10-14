"""Immutable evaluator/interface snapshots and source-drift detection."""
import hashlib
import json
from pathlib import Path
import shutil


def source_files(root):
    root=Path(root)
    files=[]
    for folder in ['inference','inference/shared','training-env','kernel-env/kernel_env',
                   'kernel-env/kernel_env/starters','kernel-env/suite','suite']:
        files.extend((root/folder).glob('*.py'))
    for relative in ['inference/tasks.json','inference/model.json','training-env/data/manifest.json']:
        if (root/relative).exists():files.append(root/relative)
    return sorted(set(files))


def hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(root)}


def snapshot(root, destination):
    root=Path(root);destination=Path(destination)
    manifest=hashes(root)
    for name,digest in manifest.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'source changed during snapshot: {name}')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
