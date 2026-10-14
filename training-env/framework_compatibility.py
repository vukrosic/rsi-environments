"""Exercise shared Verifiers 0.4.0 on immutable native source, locally only."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import verifiers.v1 as vf

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('train-smoke','heldout-smoke','null'),required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    assert importlib.metadata.version('verifiers')=='0.4.0'
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    frozen=output/'source'
    # Only needed native training/scoring and shared helper code; no model/data copies.
    for directory in (HERE,HERE.parent/'suite',HERE.parent/'suite/integration/rsi_coding_suite'):
        for source in directory.glob('*.py'):
            target=frozen/source.relative_to(HERE.parent)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    shared=HERE.parent/'suite/integration/smoke.py'
    spec=importlib.util.spec_from_file_location('shared_framework_smoke',shared)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_resolve=vf.resolve_env_config
    def resolve(config):
        harness=config['agent']['harness']
        harness['checkout']=str(frozen)
        harness['data_dir']=str(output/'absent-data') if args.case=='null' else str(HERE/'data')
        return original_resolve(config)
    module.vf.resolve_env_config=resolve # in this process only; shared source is read-only
    try:
        asyncio.run(module.run(SimpleNamespace(output=str(output/'episodes'),environments=['training'],
                    split='heldout' if args.case=='heldout-smoke' else 'train')))
    finally:module.vf.resolve_env_config=original_resolve
    summary=json.loads((output/'episodes/summary.json').read_text())
    result=summary['results'][0];native=result['native'][0]
    assert result['framework_calls']>=1 and result['trainable']==[False]
    assert native['eligible'] is False
    assert 'environment source changed during rollout' not in native['exclusion_reasons']
    if args.case=='null':
        assert native['truncated'] and native['terminal_reward'] is None and native['record']['status']=='blocked'
    else:
        assert native['terminated'] and native['record']['status']=='ok'
        expected='final' if args.case=='heldout-smoke' else 'dev'
        assert native['record']['metrics']['split']==expected
    receipt={'status':'ok','case':args.case,'verifiers':'0.4.0','native_status':native['record']['status'],
             'native_terminal':native['terminated'],'native_truncated':native['truncated'],
             'reward':native['terminal_reward'],'eligible':False,'framework_agent_trainable':False,
             'framework_calls':result['framework_calls'],'exclusion_reasons':native['exclusion_reasons'],
             'framework_ok':result['framework_ok'],'scoring_errors':result['scoring_errors'],
             'expected_scoring_exclusion':True,'outer_policy_weights_trained':False,
             'learned_improvement_verified':False,
             'shared_smoke_sha256':hashlib.sha256(shared.read_bytes()).hexdigest(),
             'frozen_source_sha256':{str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='frozen_source_sha256'}))

if __name__=='__main__':main()
