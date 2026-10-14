"""Framework-loader and harness tests; native bridge replaced by synthetic fixtures."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import verifiers.v1 as vf
from rsi_coding_suite import adapter


def test_actual_typed_framework_loader():
    config = vf.resolve_env_config({'taskset':{'id':'rsi-coding-suite'},
                                   'agent':{'runtime':{'type':'subprocess'}}})
    env = adapter.load_environment(config)
    assert isinstance(env, adapter.CodingEnv)
    assert isinstance(env.taskset, adapter.CodingTaskset)
    assert isinstance(env._harnesses['agent'], adapter.CodingHarness)
    train = list(env.taskset)
    heldout = list(vf.load_taskset(adapter.CodingConfig(id='rsi-coding-suite',split='heldout')))
    assert {t.data.id for t in train}.isdisjoint(t.data.id for t in heldout)
    assert heldout[-1].data.kernel_task=='attention'
    assert heldout[1].data.seed>=10000
    with pytest.raises(TypeError): adapter.load_environment(None)


class SyntheticBridge:
    created=[]
    @classmethod
    async def start(cls, config, data):
        obj=cls()
        obj.folder=Path(config.output_dir)/str(len(cls.created))
        obj.folder.mkdir()
        obj.actions=[]
        obj.closed=False
        cls.created.append(obj)
        return obj, {'starter_code':'# synthetic source'}
    async def exchange(self, action, timeout):
        self.actions.append(action)
        return {'observation':{'status':'ok'},'terminated':True,'truncated':False,
                'reward':0.6,'record':{'status':'ok','reward':0.6}}
    async def close(self): self.closed=True


def launch_fixture(monkeypatch,tmp_path,texts,horizon=2):
    class LocalClientFixture:
        def __init__(self,**kwargs): self.chat=SimpleNamespace(completions=self)
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def create(self,**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(texts)))])
    monkeypatch.setattr(adapter,'Bridge',SyntheticBridge)
    monkeypatch.setattr(adapter,'AsyncOpenAI',LocalClientFixture)
    return adapter.CodingHarness(adapter.CodingHarnessConfig(id='rsi-coding-suite',
        output_dir=str(tmp_path),horizon=horizon,profile='standard',compatibility_only=False))


def test_malformed_budget_and_terminal_rejection(monkeypatch,tmp_path):
    harness=launch_fixture(monkeypatch,tmp_path,iter(['broken','broken']))
    trace=SimpleNamespace(info={},agent=SimpleNamespace(trainable=True))
    data=adapter.CodingData(prompt='fixture',environment='training',split='train',seed=0)
    asyncio.run(harness.launch(SimpleNamespace(model='fixture'),trace,None,'fixture','fixture',{},data))
    assert trace.info['rsi']['policy_turns']==2
    assert trace.info['rsi']['terminated']
    assert trace.info['rsi']['terminal_reward']==0
    assert trace.info['rsi']['eligible']
    assert trace.info['rsi']['record']['provenance']['measurement'].startswith('none')
    assert SyntheticBridge.created[-1].actions==[]
    assert SyntheticBridge.created[-1].closed


def test_last_turn_forces_submit_and_fresh_episode(monkeypatch,tmp_path):
    action=json.dumps({'type':'test','code':'# fixture source'})
    harness=launch_fixture(monkeypatch,tmp_path,iter([action,action]),horizon=1)
    data=adapter.CodingData(prompt='fixture',environment='kernel',split='heldout',seed=900000001)
    traces=[]
    for _ in range(2):
        trace=SimpleNamespace(info={},agent=SimpleNamespace(trainable=True))
        asyncio.run(harness.launch(SimpleNamespace(model='fixture'),trace,None,'fixture','fixture',{},data))
        traces.append(trace)
        assert not trace.agent.trainable
        assert SyntheticBridge.created[-1].actions[0]['type']=='submit'
    assert traces[0].info['rsi']['episode_dir']!=traces[1].info['rsi']['episode_dir']


def test_immutable_source_snapshot_and_drift(tmp_path):
    from rsi_coding_suite.evidence import snapshot, hashes
    root=tmp_path/'checkout';(root/'inference').mkdir(parents=True)
    source=root/'inference/rl_environment.py';source.write_text('# synthetic interface v1')
    saved=snapshot(root,tmp_path/'snapshot')
    source.write_text('# synthetic interface v2')
    assert hashes(root)!=saved
    assert (tmp_path/'snapshot/inference/rl_environment.py').read_text()=='# synthetic interface v1'
