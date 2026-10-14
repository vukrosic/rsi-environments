import json
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tensor_hashes import state_change_evidence

class TensorEvidenceTests(unittest.TestCase):
    def test_reviewer_large_finite_update_serializes(self):
        for value in (1e200,1e-200):
            with self.subTest(value=value):
                evidence=state_change_evidence({'x':torch.zeros(1,dtype=torch.float64)},
                    {'x':torch.tensor([value],dtype=torch.float64)})
                self.assertEqual(evidence['state_update_l2'],value)
                self.assertEqual(evidence['changed_state_tensors'],1)
                json.dumps(evidence,allow_nan=False)

    def test_scaled_norm_matches_known_vector_across_tensors(self):
        before={k:torch.zeros(1,dtype=torch.float64) for k in ('a','b')}
        after={'a':torch.tensor([3e200],dtype=torch.float64),'b':torch.tensor([4e200],dtype=torch.float64)}
        evidence=state_change_evidence(before,after)
        self.assertTrue(math.isclose(evidence['state_update_l2'],5e200,rel_tol=1e-15))
        self.assertEqual(evidence['changed_state_tensors'],2)

    def test_unrepresentable_norm_is_explicit_null(self):
        cases=[({'x':torch.tensor([-1e308],dtype=torch.float64)}, {'x':torch.tensor([1e308],dtype=torch.float64)}),
               ({'x':torch.zeros(4,dtype=torch.float64)}, {'x':torch.full((4,),1e308,dtype=torch.float64)})]
        for before,after in cases:
            with self.subTest(after=after):
                evidence=state_change_evidence(before,after)
                self.assertIsNone(evidence['state_update_l2'])
                self.assertIn('finite float64',evidence['state_update_l2_reason'])
                json.dumps(evidence,allow_nan=False)

if __name__=='__main__': unittest.main()
