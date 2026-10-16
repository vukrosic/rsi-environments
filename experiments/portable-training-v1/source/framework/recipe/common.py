"""Private recipe experiment I/O and fixed-input identities. No ML imports."""
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
MODEL_MANIFEST_SHA256 = '02b6e0c1547283a267dbe05d40dbbf88aeff84f0474e7b00c5dc1f3a6e8a3b40'
OLD_BASE_WEIGHTS_SHA = '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
DATA = {'manifest.json': '5c39c9dd341cce0319cf3e47425621398d04da1d8661ee82b5681d82cb4472e2',
        'dev.npy': '9a708001cb0a88c6c7dd73bb77466fd7c66ab36e562e0b285874d910463543ea',
        'final.npy': '379555509c7ab2724b444c572d0c909f6e9baa79e70f189d51f730a11a59b6a8',
        'tok.json': '73c4400fb56436893dea14ca7867d76291389444aaddc081ff2b272d0a1f69f3',
        'train.npy': '07e9240dd8e1bd232561d6c337363c3ee3ffa75f2ca206c01c0408059ecb0047'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def path(value):
    result = Path(value)
    require(result.is_absolute() and result == result.resolve()
            and not any(p.is_symlink() for p in (result, *result.parents)), 'canonical nonsymlink path required')
    return result


def read(filename, cap=1048576):
    filename = path(filename)
    require(filename.is_file() and filename.stat().st_size <= cap, 'bounded regular metadata required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate metadata key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite metadata')
    result = json.loads(filename.read_bytes(), object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(result, dict), 'metadata object required')
    return result


def sync_dir(folder):
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(filename, value):
    filename = path(filename)
    with filename.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(filename.parent)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_files(root, pins):
    root = path(root)
    require(isinstance(pins, dict) and 1 <= len(pins) <= 256, 'bounded file map required')
    for name, pin in pins.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative pinned member required')
        item = path(root / name)
        require(item.is_file() and item.is_relative_to(root) and item.stat().st_size == pin['bytes'],
                'immutable input size differs: ' + name)
        if 'sha256' in pin:
            require(sha(item) == pin['sha256'], 'immutable input differs: ' + name)
        else:
            digest = hashlib.sha1(b'blob ' + str(pin['bytes']).encode() + b'\0')
            with item.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1048576), b''):
                    digest.update(chunk)
            require(digest.hexdigest() == pin['git_blob_sha1'], 'official git blob differs: ' + name)


def runtime():
    result = {}
    for name in ('torch', 'numpy', 'transformers', 'tokenizers', 'safetensors'):
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None, 'missing existing dependency: ' + name)
        origin = path(Path(spec.origin).resolve())
        result[name] = dict(version=importlib.metadata.version(name), path=str(origin), sha256=sha(origin))
    return result


def load_config(filename, expected):
    require(sha(filename) == expected, 'experiment configuration changed')
    cfg = read(filename)
    require(cfg['schema'] == 'recipe-grpo-experiment-v1' and cfg['model_id'] == MODEL_ID
            and cfg['group_size'] == 8 and cfg['rank'] == 16 and cfg['dtype'] == 'float32', 'wrong experiment')
    require(cfg['model_revision'] == MODEL_REVISION
            and cfg['model_manifest_sha256'] == MODEL_MANIFEST_SHA256, 'exact Instruct revision required')
    require(cfg['train_seed'] not in cfg['eval_seeds'] and len(set(cfg['eval_seeds'])) == 4,
            'fixed training seed and four distinct held-out seeds required')
    require(runtime() == cfg['runtime'], 'existing runtime changed')
    return cfg


def inputs(cfg):
    require(sha(HERE / 'SOURCE-MANIFEST.json') == cfg['source_manifest_sha256'],
            'source manifest changed since campaign initialization')
    verify_files(HERE, read(HERE / 'SOURCE-MANIFEST.json')['files'])
    native = path(cfg['native_root'])
    native_pins = read(HERE / 'NATIVE-SOURCE-PINS.json')['files']
    require(len(native_pins) == 53, 'full fixed native source required')
    verify_files(native, native_pins)
    model_manifest = read(cfg['model_manifest'])
    require(sha(cfg['model_manifest']) == cfg['model_manifest_sha256']
            and model_manifest['model_id'] == MODEL_ID and model_manifest['revision'] == cfg['model_revision'],
            'exact Instruct model identity required')
    require(model_manifest['files']['model.safetensors']['sha256'] != OLD_BASE_WEIGHTS_SHA,
            'old base model is not the requested Instruct policy')
    verify_files(cfg['model'], model_manifest['files'])
    for name, expected in DATA.items():
        require(sha(path(cfg['data']) / name) == expected, 'fixed prepared data changed: ' + name)
    require(sha(path(cfg['data']).parent / 'workload.json') ==
            '14066318b4303324268f6574afef679a8fba2314dd7246260832e3339003a1fe', 'fixed workload changed')
    require(runtime() == cfg['runtime'], 'runtime identities changed')


def native_path(cfg):
    import sys
    root = path(cfg['native_root'])
    sys.path[:0] = [str(root / 'training-env'), str(root)]
    return root


def prompt():
    return ('Choose an optimizer recipe for a fixed 5,063,424-parameter GPT trained for exactly 500 calls, '
            'batch32 context256 on fixed TinyStories train text, graded on dev text. Only the recipe changes. '
            'Return exactly one JSON object, no markdown or explanation. All six fields are required: '
            'lr in[0.00001,0.01], warmup_frac in[0,0.3], weight_decay in[0,0.3], '
            'schedule one of cosine/linear/constant, grad_clip in[0.1,10], beta2 in[0.9,0.999]. '
            'Example valid format: {"lr":0.001,"warmup_frac":0.03,"weight_decay":0.1,'
            '"schedule":"cosine","grad_clip":1.0,"beta2":0.95}. '
            'Choose values to maximize exp(-dev_loss).')
