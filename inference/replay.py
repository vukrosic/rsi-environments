"""Verify and recheck saved tensors. No model execution or timing reward."""
import argparse
import json
from pathlib import Path
import torch
from assets import sha
from scorer import compare

def recorded_file(directory, name, digest):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f'artifact path escapes run: {name}')
    if not isinstance(digest,str) or sha(path) != digest:
        raise ValueError(f'artifact hash mismatch: {name}')
    return path

def replay(directory):
    directory = Path(directory).resolve()
    result = {'schema_version':1, 'kind':'saved_correctness_replay',
        'artifact_dir':str(directory), 'model_executed':False, 'timing_valid':False,
        'reward':None, 'status':'error', 'ok':False}
    try:
        receipt = directory / 'result.json'
        result['source_result_sha256'] = sha(receipt)
        original = json.loads(receipt.read_text())
        result['original_status'] = original['status']
        result['original_source_sha256'] = original['source_sha256']
        for name, digest in original['evaluator_hashes'].items():
            recorded_file(directory,name,digest)
        recorded_file(directory,'submission.py',original['source_sha256'])
        recorded_file(directory,'inputs.pt',original['inputs_sha256'])
        artifacts = {}
        for name in ('baseline_before','candidate','baseline_after'):
            path = recorded_file(directory,name+'.pt',original['workers'][name]['artifact_sha256'])
            artifacts[name] = torch.load(path,map_location='cpu',weights_only=True)
        result['artifact_hashes_verified'] = True
        oracle = compare(artifacts['baseline_before'],artifacts['baseline_after'])
        candidate = compare(artifacts['baseline_before'],artifacts['candidate'])
        result['oracle_comparison'] = oracle
        result['candidate_comparison'] = candidate
        result['oracle_consistent'] = oracle['ok']
        result['candidate_matches_initial_reference'] = candidate['ok']
        result['ok'] = oracle['ok'] and candidate['ok']
        result['status'] = ('passed' if candidate['ok'] else 'invalid') if oracle['ok'] else 'error'
        if not oracle['ok']:
            result['error'] = 'saved oracle disagreement: '+oracle['error']
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,help='complete run directory containing result.json and raw tensors')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = replay(args.run)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))
