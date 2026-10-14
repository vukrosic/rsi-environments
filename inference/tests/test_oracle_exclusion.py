"""Actual scorer -> normalized record -> RL transition, CPU fixture workers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorer
import rl_environment
from rl_environment import CodingAgentEnvironment

def artifact(duration, offset=0.0):
    rows = []
    for length in scorer.LENGTHS:
        logits = torch.tensor([[0.0, 1.0]]) + offset
        rows.append(dict(key=str(length), length=length, first=logits,
            greedy=logits.unsqueeze(0), forced=logits.unsqueeze(0),
            tokens=torch.tensor([[[1]]]), prefill_s=duration, decode_s=duration))
    return dict(rows=rows, engine_init_s=0.001, warmup_s=0.001,
                mps_memory={}, torch_threads=2)

def fixture_workers(offset=0.0, candidate_offset=0.0):
    def worker(source, inputs, output, device, deadline):
        shift = offset if output.stem == 'baseline_after' else (
            candidate_offset if output.stem == 'candidate' else 0.0)
        torch.save(artifact(0.025 if output.stem == 'candidate' else 0.1, shift), output)
        return {'conditions':[], 'artifact_sha256':scorer.sha(output)}
    return worker

class OracleExclusionTests(unittest.TestCase):
    def transition(self, oracle_offset, candidate_offset=0.0, max_steps=1):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(rl_environment,'evaluate',side_effect=scorer._evaluate_with_errors), \
             patch.object(scorer,'local_model'), \
             patch.object(scorer,'machine',return_value={'blockers':[]}), \
             patch.object(scorer,'make_inputs',return_value={'cases':[],'decode_steps':8,'task_id':'fixture-train'}), \
             patch.object(scorer,'child',side_effect=fixture_workers(oracle_offset,candidate_offset)):
            env = CodingAgentEnvironment(root, max_steps=max_steps)
            observation = env.reset(split='train', seed=19)
            transition = env.step({'type':'submit','code':observation['starter_code']})
            if oracle_offset:
                raw = json.loads(next(env.episode.glob('*/result.json')).read_text())
                self.assertIsNone(raw['reward'])
                self.assertEqual(raw['status'],'failed')
                self.assertFalse(raw['timing_valid'])
                self.assertIn('baseline nondeterminism',raw['error'])
                self.assertEqual(raw['metrics']['candidate_matches_initial_reference'],candidate_offset == 0)
                self.assertFalse(raw['metrics']['baseline_comparison']['ok'])
            return transition

    def test_consistent_oracle_keeps_correct_reward(self):
        transition = self.transition(0)
        self.assertEqual(transition['info']['record']['status'],'ok')
        self.assertAlmostEqual(transition['reward'],0.8)
        self.assertTrue(transition['info']['trainable'])

    def test_disagreement_excludes_matching_candidate(self):
        transition = self.transition(1)
        self.assertIsNone(transition['reward'])
        self.assertFalse(transition['info']['trainable'])
        self.assertTrue(transition['terminated'])
        self.assertEqual(transition['info']['record']['status'],'error')

    def test_disagreement_ends_nonterminal_episode(self):
        transition = self.transition(1,max_steps=3)
        self.assertTrue(transition['terminated'])
        self.assertIsNone(transition['reward'])
        self.assertFalse(transition['info']['trainable'])

    def test_disagreement_takes_precedence_over_bad_candidate(self):
        transition = self.transition(1,candidate_offset=2)
        self.assertEqual(transition['info']['record']['status'],'error')
        self.assertIsNone(transition['reward'])
        self.assertFalse(transition['info']['trainable'])

    def test_bad_candidate_with_consistent_oracle_still_zero(self):
        transition = self.transition(0,candidate_offset=2)
        self.assertEqual(transition['info']['record']['status'],'invalid')
        self.assertEqual(transition['reward'],0)
        self.assertTrue(transition['info']['trainable'])

if __name__ == '__main__':
    unittest.main()
