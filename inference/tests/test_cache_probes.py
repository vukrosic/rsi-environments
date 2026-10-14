"""Handcrafted context-sensitive smoke oracle, not pretrained-model evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cache_probes import ContractViolation,inputs,run
from solution import Engine
from scorer import compare
from test_oracle_exclusion import artifact

class ContextModel:
    def full(self,ids):
        position=torch.arange(1,ids.shape[1]+1).view(1,-1)
        value=(ids*position).sum(-1,keepdim=True)+ids.shape[1]*3
        vocabulary=torch.arange(32).view(1,-1)
        return -((value%32)-vocabulary).float().square()
    def __call__(self,input_ids,past_key_values=None,**kwargs):
        context=input_ids.clone() if past_key_values is None else torch.cat([past_key_values,input_ids],dim=1)
        return SimpleNamespace(logits=self.full(context).unsqueeze(1),past_key_values=context)

class StaleCache(Engine):
    def prefill(self,ids):
        result=self.model(input_ids=ids,past_key_values=self.cache)
        self.cache=result.past_key_values
        return result.logits[:,-1,:]

class CacheProbeTests(unittest.TestCase):
    def probe_inputs(self):
        return inputs('0123456789abcdef0123456789abcdef',32)
    def good(self):
        model=ContextModel()
        return run(Engine(model),self.probe_inputs(),'cpu',oracle=model.full)
    def test_cache_reset_prefix_branches_and_batch_changes(self):
        probes=self.good()
        self.assertEqual(len(probes),13)
        values={p['key']:p for p in probes}
        self.assertTrue(torch.equal(values['request-a']['logits'],values['batch-one-after-two']['logits']))
        self.assertEqual(values['batch-two-step']['logits'].shape,(2,32))
        self.assertEqual(values['shorter-prefix-reset']['context'].shape,(1,3))
        self.assertFalse(torch.equal(values['branch-left-1']['context'],values['branch-right-1']['context']))
    def test_stale_cache_fails_independent_full_context_oracle(self):
        model=ContextModel()
        with self.assertRaises(AssertionError):
            run(StaleCache(model),self.probe_inputs(),'cpu',oracle=model.full)
    def test_probe_output_shape_uses_explicit_contract_failure(self):
        model=ContextModel()
        class WrongShape(Engine):
            def prefill(self,ids):return super().prefill(ids).unsqueeze(0)
        with self.assertRaises(ContractViolation):
            run(WrongShape(model),self.probe_inputs(),'cpu')
    def test_probe_input_mutation_uses_explicit_contract_failure(self):
        model=ContextModel()
        class MutatingInput(Engine):
            def prefill(self,ids):
                ids[0,0]+=1
                return super().prefill(ids)
        with self.assertRaises(ContractViolation):
            run(MutatingInput(model),self.probe_inputs(),'cpu')
    def test_wrong_probes_zero_even_when_timed_rows_match(self):
        reference=artifact(.1);candidate=artifact(.1)
        reference['cache_probes']=self.good();candidate['cache_probes']=self.good()
        candidate['cache_probes'][3]['logits'][0,0]+=5
        verdict=compare(reference,candidate)
        self.assertFalse(verdict['ok']);self.assertEqual(verdict['reward'],0)
    def test_missing_probes_and_changed_context_rejected(self):
        reference=artifact(.1);reference['cache_probes']=self.good()
        for changed in (False,True):
            candidate=artifact(.1)
            if changed:
                candidate['cache_probes']=self.good()
                candidate['cache_probes'][0]['context'][0,0]+=1
            self.assertFalse(compare(reference,candidate)['ok'])

if __name__=='__main__':unittest.main()
