"""Synthetic receipt/schema fixtures; no benchmark-validity claims."""
import json
import pytest
from rsi_coding_suite.protocol import eligibility, parse_action, policy_view


def receipt(status='ok', reward=0.4, **updates):
    return dict({'record':{'status':status,'reward':reward}, 'reward':reward,
                 'terminated':True,'truncated':False,'info':{}}, **updates)


def eligible(t, split='train', compatibility_only=False, profile='standard'):
    return eligibility(t, split=split, compatibility_only=compatibility_only, profile=profile)[0]


@pytest.mark.parametrize('status,reward',[('blocked',None),('error',None),('ok',None),('ok',float('nan')),('ok',float('inf')),('ok',True),('ok',-0.1),('ok',1.1)])
def test_failed_receipts_never_train(status,reward):
    assert not eligible(receipt(status,reward))


def test_train_gate_and_heldout_smoke_terminal_separation():
    assert eligible(receipt())
    assert eligible(receipt('invalid',0.0))
    assert not eligible(receipt(),split='heldout')
    assert not eligible(receipt(),compatibility_only=True)
    assert not eligible(receipt(),profile='smoke')
    assert not eligible(receipt(terminated=False))
    assert not eligible(receipt(truncated=True))
    assert not eligible(receipt(info={'trainable':False}))
    assert not eligible(receipt(info={'exclude_from_rl':True}))
    assert not eligible({'reward':0.5,'terminated':True})


@pytest.mark.parametrize('text',['', 'not json','[]','null','{}', '{"type":"submit","code":"x","extra":1}', '{"type":"submit","code":""}', '{"type":"submit","code":17}', '{"type":"shell","code":"x"}', '{"type":"submit","type":"test","code":"x"}'])
def test_malformed_actions(text):
    with pytest.raises((ValueError,TypeError)): parse_action(text,'training')


def test_code_string_limits_and_types():
    assert parse_action(json.dumps({'type':'submit','code':'print(1)'}),'training')['code']=='print(1)'
    assert parse_action(json.dumps({'type':'edit','code':'x'}),'kernel')['type']=='edit'
    with pytest.raises(ValueError): parse_action(json.dumps({'type':'edit','code':'x'}),'training')
    with pytest.raises(ValueError): parse_action(json.dumps({'type':'test','code':'x'*65537}),'inference')


def test_operator_paths_not_policy_context():
    assert policy_view({'code':'source','candidate':'/private/candidate','feedback':{
        'reward':0.4,'provenance':{'source_snapshot':'private'}}}) == {'code':'source','feedback':{'reward':0.4}}


def test_native_partition_profile_and_reward_conflicts():
    assert not eligible(receipt(info={'task_split':'heldout'}))
    t=receipt();t['record']['profile']='smoke'
    assert not eligible(t)
    t=receipt();t['record']['reward']=0.9
    assert not eligible(t)
    t=receipt();t['record'].update(environment='kernel',task='attention')
    assert not eligible(t)
    t=receipt();t['record'].update(environment='training')
    t['info']['scored_data_split']='final'
    assert not eligible(t)
    t=receipt();t['record']['metrics']={'timing_valid':False}
    assert not eligible(t)
