"""Replay verifies artifacts and never presents old correctness as fresh reward."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assets import sha
from replay import replay
from test_oracle_exclusion import artifact

class ReplayTests(unittest.TestCase):
    def fixture(self,root,oracle_offset=0,candidate_offset=0):
        (root/'submission.py').write_text('class Engine: pass\n')
        torch.save({'cases':[]},root/'inputs.pt')
        workers = {}
        for name, offset in [('baseline_before',0),('candidate',candidate_offset),('baseline_after',oracle_offset)]:
            torch.save(artifact(0.1,offset),root/(name+'.pt'))
            workers[name] = {'artifact_sha256':sha(root/(name+'.pt'))}
        receipt = {'status':'diagnostic_pass','source_sha256':sha(root/'submission.py'),
            'evaluator_hashes':{'submission.py':sha(root/'submission.py')},
            'inputs_sha256':sha(root/'inputs.pt'),'workers':workers}
        (root/'result.json').write_text(json.dumps(receipt))

    def test_replay_correctness_without_reward_or_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root)
            result=replay(root)
            self.assertEqual(result['status'],'passed')
            self.assertTrue(result['artifact_hashes_verified'])
            self.assertTrue(result['ok'])
            self.assertIsNone(result['reward'])
            self.assertFalse(result['model_executed'])

    def test_oracle_disagreement_remains_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root,oracle_offset=1)
            result=replay(root)
            self.assertEqual(result['status'],'error')
            self.assertTrue(result['candidate_matches_initial_reference'])
            self.assertIsNone(result['reward'])

    def test_wrong_candidate_remains_invalid_but_no_new_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root,candidate_offset=1)
            result=replay(root)
            self.assertEqual(result['status'],'invalid')
            self.assertIsNone(result['reward'])

    def test_modified_source_or_output_refused(self):
        for name in ('submission.py','candidate.pt','inputs.pt'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); self.fixture(root)
                with (root/name).open('ab') as file: file.write(b'changed')
                result=replay(root)
                self.assertEqual(result['status'],'error')
                self.assertIn('hash mismatch',result['error'])
                self.assertIsNone(result['reward'])

if __name__ == '__main__':
    unittest.main()
