"""Bounded CPU adversarial tests; no standard training or accelerators."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks import InvalidCandidate, check_model, check_logits, check_causal
from data import load_data, MANIFEST_VERSION
from protocol import MAX_PARAMS, VOCAB


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(VOCAB))

    def forward(self, x):
        return self.bias.expand(*x.shape, VOCAB)


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x = torch.arange(16).reshape(2, 8)

    def test_valid_minimal_model(self):
        m = Tiny()
        self.assertEqual(check_model(m), VOCAB)
        check_causal(m, self.x)

    def test_wrong_logits_shapes(self):
        for shape in ((2, 8), (2, 8, 4095), (1, 8, VOCAB)):
            with self.subTest(shape=shape), self.assertRaises(InvalidCandidate):
                check_logits(torch.zeros(shape), self.x)

    def test_nonfinite_logits_and_weights(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                m = Tiny()
                with torch.no_grad():
                    m.bias[0] = value
                with self.assertRaises(InvalidCandidate):
                    check_model(m)
                with self.assertRaises(InvalidCandidate):
                    check_logits(m(self.x), self.x)

    def test_parameter_budget_includes_frozen_parameters(self):
        m = Tiny()
        m.extra = torch.nn.Parameter(torch.zeros(MAX_PARAMS), requires_grad=False)
        with self.assertRaises(InvalidCandidate):
            check_model(m)

    def test_future_token_leak(self):
        class Leaky(Tiny):
            def forward(self, x):
                return super().forward(x) + x.sum(dim=1)[:, None, None]
        with self.assertRaisesRegex(InvalidCandidate, 'future'):
            check_causal(Leaky(), self.x)

    def test_sequence_length_leak(self):
        class LengthLeak(Tiny):
            def forward(self, x):
                return super().forward(x) + x.shape[1]
        with self.assertRaisesRegex(InvalidCandidate, 'sequence length'):
            check_causal(LengthLeak(), self.x)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'version': MANIFEST_VERSION, 'kind': 'synthetic',
                         'vocab_size': VOCAB, 'counts': {}, 'sha256': {}}
        for i, split in enumerate(('train', 'dev', 'final')):
            path = self.root / (split + '.npy')
            np.save(path, np.arange(1024, dtype=np.uint16) + i)
            self.manifest['counts'][split] = 1024
            self.manifest['sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_smoke_load(self):
        train, dev, final, metadata = load_data(self.root, 'smoke')
        self.assertFalse(np.array_equal(dev, final))
        self.assertEqual(metadata['kind'], 'synthetic')

    def test_standard_rejects_synthetic(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            load_data(self.root, 'standard')

    def test_missing_assets(self):
        (self.root / 'dev.npy').unlink()
        with self.assertRaises(ValueError):
            load_data(self.root, 'smoke')

    def test_candidate_reported_loss_cannot_spoof_reward(self):
        from evaluate import evaluate
        candidate = self.root / 'spoof.py'
        candidate.write_text('''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.shift.expand(*x.shape, 4096)
def build_model(): return Model()
def build_optimizer(model): return None
def train_step(model, opt, x, y, frac):
    with torch.no_grad(): model.shift.add_(0.01)
    return -1000000.0
''')
        record = evaluate(candidate, self.root, self.root / 'run', device='cpu')
        self.assertEqual(record['status'], 'ok', record)
        self.assertAlmostEqual(record['metrics']['held_out_loss'], math.log(VOCAB), places=5)
        self.assertEqual(record['metrics']['steps'], 3)
        self.assertTrue((self.root / 'run' / 'weights.pt').is_file())

    def test_nan_reported_loss_ignored(self):
        from evaluate import evaluate
        candidate = self.root / 'nan_report.py'
        candidate.write_text('''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return self.shift.expand(*x.shape, 4096)
def build_model(): return Model()
def build_optimizer(model): return None
def train_step(model, opt, x, y, frac):
    with torch.no_grad(): model.shift.add_(0.01)
    return float('nan')
''')
        from interaction import TrainingEnv
        env = TrainingEnv(self.root / 'rl-session', self.root, device='cpu')
        result = env.step({'code': candidate.read_text(), 'final': True})
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertIsInstance(result['reward'], float)
        self.assertEqual(result['info']['scored_data_split'], 'dev')
        self.assertEqual((self.root / 'rl-session' / 'actions' / 'final.py').read_text(), candidate.read_text())
        record = result['record']
        self.assertEqual(record['status'], 'ok', record)
        self.assertAlmostEqual(record['metrics']['held_out_loss'], math.log(VOCAB), places=5)

    def test_syntax_invalid_candidate_is_zero_reward(self):
        from interaction import TrainingEnv
        env = TrainingEnv(self.root / 'syntax-session', self.root)
        result = env.step({'code': 'def broken(:', 'final': True})
        self.assertEqual(result['record']['status'], 'invalid', result)
        self.assertEqual(result['reward'], 0.0)
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertFalse(result['info']['exclude_from_rl'])

    def test_hash_tampering(self):
        with (self.root / 'train.npy').open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_data(self.root, 'smoke')

    def test_identical_dev_final_rejected(self):
        (self.root / 'final.npy').write_bytes((self.root / 'dev.npy').read_bytes())
        self.manifest['sha256']['final.npy'] = self.manifest['sha256']['dev.npy']
        self.write_manifest()
        with self.assertRaises(ValueError):
            load_data(self.root, 'smoke')


class InteractionTests(unittest.TestCase):
    def test_persistent_budget_and_single_final(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = TrainingEnv(root / 'session', root / 'data')
            record = {'status': 'ok', 'metrics': {'held_out_loss': 8.0}, 'reward': 0.1}
            with patch('interaction.evaluate', return_value=record) as evaluate:
                for i in range(10):
                    self.assertFalse(env.step('candidate.py')['terminated'])
                    self.assertEqual(env.reset()['trials_remaining'], 9 - i)
                reopened = TrainingEnv(root / 'session', root / 'data')
                self.assertEqual(reopened.reset()['trials_remaining'], 0)
                with self.assertRaisesRegex(RuntimeError, 'budget'):
                    reopened.step('candidate.py')
                self.assertTrue(reopened.final('candidate.py')['terminated'])
                self.assertEqual(evaluate.call_args.args[4], 'dev')
                for action in (reopened.step, reopened.final):
                    with self.assertRaisesRegex(RuntimeError, 'finalized'):
                        action('candidate.py')
                self.assertEqual(evaluate.call_count, 11)

    def test_failures_consume_reserved_trial(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            env = TrainingEnv(Path(directory) / 'session', directory, max_trials=1)
            with patch('interaction.evaluate', side_effect=RuntimeError('crash')):
                with self.assertRaisesRegex(RuntimeError, 'crash'):
                    env.step('candidate.py')
            self.assertEqual(env.reset()['trials_remaining'], 0)

    def test_code_action_snapshots_and_split_rewards(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            for split, seed, expected in (('train', 0, 'dev'), ('heldout', 10000, 'final')):
                env = TrainingEnv(Path(directory) / split, directory, task_split=split, seed=seed)
                obs = env.reset()
                self.assertEqual(obs['terminal_reward_split'], expected)
                self.assertIn('action_schema', obs)
                self.assertIn('protocol', obs)
                self.assertIn('starter_code', obs)
                record = {'status': 'ok', 'metrics': {'held_out_loss': 8.0}, 'reward': 0.25}
                with patch('interaction.evaluate', return_value=record) as evaluate:
                    intermediate = env.step({'code': '# source fixture'})
                    self.assertIsNone(intermediate['reward'])
                    self.assertFalse(intermediate['terminated'])
                    self.assertEqual(intermediate['observation']['measured_dev_reward'], 0.25)
                    result = env.step({'code': '# final fixture', 'final': True})
                    self.assertEqual(result['reward'], 0.25)
                    self.assertTrue(result['terminated'])
                    self.assertFalse(result['truncated'])
                    self.assertEqual(result['info']['scored_data_split'], expected)
                    self.assertEqual(evaluate.call_args.args[4], expected)
                    source = Path(evaluate.call_args.args[0])
                    self.assertEqual(source.read_text(), '# final fixture')
                    self.assertEqual((source.parent / 'dev-01.py').read_text(), '# source fixture')

    def test_strict_action_validation_preserves_budget(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            env = TrainingEnv(Path(directory) / 'session', directory)
            for action in (None, 42, {}, {'code': 1}, {'code': ''}, {'code': 'x', 'final': 1},
                           {'code': 'x', 'other': True}, {'code': 'x' * 256001}):
                with self.subTest(action_type=type(action)), self.assertRaises(ValueError):
                    env.step(action)
            self.assertEqual(env.reset()['trials_remaining'], 10)

    def test_disjoint_episode_seed_ranges(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            for split, seed in (('train', 10000), ('heldout', 9999), ('train', -1), ('heldout', 20000), ('train', True)):
                with self.subTest(split=split, seed=seed), self.assertRaises(ValueError):
                    TrainingEnv(Path(directory) / 'session', directory, task_split=split, seed=seed)

    def test_infrastructure_truncation_closes_episode(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            for status in ('blocked', 'error'):
                env = TrainingEnv(Path(directory) / status, directory)
                record = {'status': status, 'metrics': {'reason': 'infrastructure fixture'}, 'reward': None}
                with patch('interaction.evaluate', return_value=record):
                    result = env.step({'code': '# fixture'})
                self.assertTrue(result['truncated'])
                self.assertTrue(result['info']['exclude_from_rl'])
                self.assertIsNone(result['reward'])
                self.assertTrue(env.reset()['finalized'])
                with self.assertRaises(RuntimeError):
                    env.step({'code': '# retry'})

    def test_invalid_budgets(self):
        from interaction import TrainingEnv
        with tempfile.TemporaryDirectory() as directory:
            for budget in (0, 11, 1.5, True):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    TrainingEnv(Path(directory) / 'session', directory, max_trials=budget)



if __name__ == '__main__':
    unittest.main()
