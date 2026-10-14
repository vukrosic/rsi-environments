"""Exercise the coordinator's adapter; no duplicate adapter or policy training.

The local provider deliberately sends syntax-invalid source, checking the current
environment boundary with no model/scoring workload. Compatibility-only episodes
must remain excluded from RL updates. The shared integration package is required.
"""
import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

def main(output):
    integration=Path(__file__).resolve().parent.parent/'suite'/'integration'
    sys.path.insert(0,str(integration))
    spec=importlib.util.spec_from_file_location('shared_smoke_verification',integration/'smoke.py')
    smoke=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    version=importlib.metadata.version('verifiers')
    if version!='0.4.0':raise ValueError('requires coordinator-pinned Verifiers 0.4.0')
    original=smoke.web.json_response
    def local_invalid_action(value,*args,**kwargs):
        if isinstance(value,dict) and 'choices' in value:
            value['choices'][0]['message']['content']=json.dumps({'type':'submit','code':'invalid syntax !'})
        return original(value,*args,**kwargs)
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    for split in ('train','heldout'):
        with patch.object(smoke.web,'json_response',side_effect=local_invalid_action):
            asyncio.run(smoke.run(argparse.Namespace(output=str(root/split),environments=['inference'],split=split)))
        summary=json.loads((root/split/'summary.json').read_text())
        result=summary['results'][0]
        native=result['native'][0]
        assert result['framework_calls'] and native['terminated']
        assert native['record']['status']=='invalid' and native['terminal_reward']==0
        assert not native['eligible'] and result['trainable']==[False]
    from assets import sha
    report={'verifiers_version':version, 'check':'current native API via actual shared framework, local scripted syntax-invalid actions',
        'model_executed':False,'policy_weights_updated':False,'learning_gain_verified':False,
        'splits':['train','heldout'],'compatibility_only':True,
        'shared_source_hashes':{name:sha(integration/name) for name in ['smoke.py','rsi_coding_suite/adapter.py','rsi_coding_suite/worker.py','rsi_coding_suite/protocol.py']},
        'native_source_hashes':{name:sha(Path(__file__).resolve().parent/name) for name in ['rl_environment.py','scorer.py','controller.py','supervisor.py','tasks.json','cache_probes.py']}}
    (root/'boundary-verification.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
