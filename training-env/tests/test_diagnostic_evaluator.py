"""Evaluator boundary receipts use synthetic CPU fixtures, never policy evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate
from prepare import prepare_synthetic
from tensor_hashes import tensor_state_sha256
from test_training_diagnostic import SOURCE
# A candidate-owned file proves the evaluator actually invoked twenty steps.
COUNTED_SOURCE = SOURCE.replace('    opt.zero_grad()', '''    from pathlib import Path
    with Path('step-calls.txt').open('a') as receipt: receipt.write('step\\n')
    opt.zero_grad()''')

class DiagnosticEvaluatorTests(unittest.TestCase):
    def test_real_children_and_checkpoint_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'source.py'
            candidate.write_text(COUNTED_SOURCE)
            prepare_synthetic(root / 'data', tokens_per_split=2048)
            record = evaluate.evaluate(candidate, root / 'data', root / 'run', diagnostic='cpu-training-v1')
            self.assertEqual(record['status'], 'ok', record)
            metrics, provenance = record['metrics'], record['provenance']
            self.assertEqual(record['profile'], 'smoke')
            self.assertFalse(metrics['policy_trainable'])
            self.assertFalse(provenance['outer_policy_trainable'])
            self.assertEqual(provenance['data_kind'], 'synthetic')
            self.assertEqual((metrics['steps'], metrics['training_tokens'], metrics['eval_tokens']), (20,5120,1024))
            self.assertEqual((root/'run'/'step-calls.txt').read_text().splitlines(), ['step']*20)
            initial = tensor_state_sha256(torch.load(root/'run'/'initial-weights.pt',weights_only=True))
            final = tensor_state_sha256(torch.load(root/'run'/'weights.pt',weights_only=True))
            self.assertEqual(initial, metrics['initial_tensor_sha256'])
            self.assertEqual(initial, metrics['reference_tensor_sha256'])
            self.assertEqual(final, metrics['final_tensor_sha256'])
            self.assertEqual(final, metrics['loaded_final_tensor_sha256'])
            self.assertNotEqual(initial, final)
            self.assertGreater(metrics['state_update_l2'], 0)
            self.assertGreater(metrics['held_out_loss'], 0)
            self.assertEqual(json.loads((root/'run'/'record.json').read_text()), record)

    def test_optional_suite_gate_excludes_native_smoke(self):
        # Standalone training source packages do not include the coordinator.
        gate_path = ROOT.parent / 'suite' / 'integration' / 'rsi_coding_suite' / 'protocol.py'
        if not gate_path.is_file():
            self.skipTest('optional coordinator integration is absent from standalone training package')
        import importlib.util
        spec = importlib.util.spec_from_file_location('diagnostic_suite_gate', gate_path)
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        record = {'environment': 'training', 'profile': 'smoke', 'status': 'ok', 'reward': 0.4,
                  'metrics': {'diagnostic': 'cpu-training-v1', 'policy_trainable': False},
                  'provenance': {'outer_policy_trainable': False}}
        transition = {'record': record, 'reward': record['reward'],
                      'terminated': True, 'truncated': False,
                      'info': {'scored_data_split': 'dev'}}
        eligible, reasons = gate.eligibility(transition, split='train', compatibility_only=False, profile='standard')
        self.assertFalse(eligible)
        self.assertIn('native smoke profile', reasons)

    def test_short_diagnostic_assets_block_before_candidate_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate,'child') as child:
            root=Path(directory)
            candidate=root/'source.py';candidate.write_text(SOURCE)
            prepare_synthetic(root/'data',tokens_per_split=1024)
            record=evaluate.evaluate(candidate,root/'data',root/'run',diagnostic='cpu-training-v1')
            self.assertEqual(record['status'],'blocked',record)
            self.assertIsNone(record['reward'])
            self.assertFalse(record['provenance']['outer_policy_trainable'])
            self.assertIn('1040 tokens',record['metrics']['reason'])
            child.assert_not_called()

    def test_rejects_profile_and_device_before_output_or_child(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate, 'child') as child, patch.object(evaluate, 'device_available') as available:
            root = Path(directory)
            for profile,device in [('standard','cpu'),('smoke','cuda'),('smoke','mps'),('smoke','cuda:0')]:
                with self.subTest(profile=profile,device=device), self.assertRaises(ValueError):
                    evaluate.evaluate(root/'missing.py',root/'missing-data',root/'run',profile=profile,device=device,diagnostic='cpu-training-v1')
                self.assertFalse((root/'run').exists())
            child.assert_not_called()
            available.assert_not_called()

    def test_independent_hash_disagreement_invalidates_each_side(self):
        for field in ['loaded_final_tensor_sha256', 'reference_tensor_sha256']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate = root/'source.py'
                candidate.write_text(SOURCE)
                prepare_synthetic(root/'data',tokens_per_split=2048)
                def fake_child(script, arguments, output, timeout):
                    if script == 'worker.py':
                        (output/'weights.pt').write_bytes(b'final')
                        (output/'initial-weights.pt').write_bytes(b'initial')
                        return {'status':'ok','initial_tensor_sha256':'initial','final_tensor_sha256':'final'}
                    result = {'status':'ok','held_out_loss':1.0,'loaded_final_tensor_sha256':'final','reference_tensor_sha256':'initial'}
                    result[field]='disagrees'
                    return result
                with patch.object(evaluate,'child',side_effect=fake_child):
                    record=evaluate.evaluate(candidate,root/'data',root/'run',diagnostic='cpu-training-v1')
                self.assertEqual(record['status'],'invalid',record)
                self.assertEqual(record['reward'],0.0)
                self.assertIn('tensor hashes disagree',record['metrics']['reason'])

if __name__ == '__main__': unittest.main()
