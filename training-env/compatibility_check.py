"""Read-only shared backend contract check; writes evidence only in training-env."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CODE='''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=torch.nn.Parameter(torch.zeros(4096))
    def forward(self,x):return self.bias.expand(*x.shape,4096)
def build_model():return Model()
def build_optimizer(model):return torch.optim.SGD(model.parameters(),lr=.01)
def train_step(model,opt,x,y,frac):
    loss=torch.nn.functional.cross_entropy(model(x).reshape(-1,4096),y.reshape(-1))
    opt.zero_grad();loss.backward();opt.step()
    return loss
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    backend=HERE.parent/'suite/integration/rsi_coding_suite/worker.py'
    protocol=backend.with_name('protocol.py')
    spec=importlib.util.spec_from_file_location('shared_policy_protocol',protocol)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    config=dict(checkout=str(HERE.parent),workspace=str(output/'episode'),environment='training',
                device='cpu',seed=0,split='train',profile='smoke',horizon=2,
                calibration=None,data_dir=str(Path(args.data).resolve()),compatibility_only=True)
    lines=[config,{'type':'test','code':CODE},{'type':'submit','code':CODE}]
    run=subprocess.run([sys.executable,str(backend)],input=''.join(json.dumps(line)+'\n' for line in lines),
                       text=True,capture_output=True,timeout=200)
    if run.returncode:raise RuntimeError(run.stderr)
    observation,feedback,terminal=[json.loads(line) for line in run.stdout.splitlines()]
    assert observation['api_version']=='training-episode-v1'
    assert not feedback['terminated'] and feedback['reward'] is None
    assert terminal['terminated'] and not terminal['truncated'] and terminal['record']['status']=='ok'
    assert terminal['info']['scored_data_split']=='dev' and terminal['info']['trainable'] is False
    eligible,reasons=mod.eligibility(terminal,split='train',compatibility_only=True,profile='smoke')
    assert not eligible and any('smoke' in reason for reason in reasons)
    receipt={'shared_backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
             'shared_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
             'profile':'smoke','device':'cpu','outer_policy_trained':False,
             'status':'ok','api_version':observation['api_version'],
             'terminal_reward':terminal['reward'],'trainable':eligible,'eligibility_reasons':reasons,
             'scope':'actual shared backend JSON-lines reset/test/submit; not a framework rollout or optimizer update'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
