import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import grader
import worker


class ExecutionStatusTests(unittest.TestCase):
    def invoke_grader(self, root, candidate, checkpoint, heldout):
        result_path = Path(root) / 'result.json'
        argv = ['grader.py', '--candidate', str(candidate), '--checkpoint', str(checkpoint),
                '--heldout', str(heldout), '--profile', 'smoke', '--device', 'cpu',
                '--result', str(result_path)]
        with patch.object(sys, 'argv', argv):
            grader.main()
        return json.loads(result_path.read_text())

    def test_missing_checkpoint_is_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            candidate = root / 'candidate.py'
            candidate.write_text('''import torch\n\ndef build_model(): return torch.nn.Linear(1, 1)\ndef build_optimizer(m): return None\ndef train_step(*a): pass\n''')
            heldout = root / 'heldout.npy'
            np.save(heldout, np.arange(100, dtype=np.int64))
            result = self.invoke_grader(root, candidate, root / 'missing.pt', heldout)
            self.assertEqual(result['status'], 'error')
            self.assertIsNone(result.get('held_out_loss'))
            self.assertEqual(result['phase'], 'checkpoint-read')

    def test_bad_candidate_syntax_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            candidate = root / 'candidate.py'
            candidate.write_text('def build_model(:\n')
            heldout = root / 'heldout.npy'
            np.save(heldout, np.arange(100, dtype=np.int64))
            result = self.invoke_grader(root, candidate, root / 'missing.pt', heldout)
            self.assertEqual(result['status'], 'invalid')
            self.assertEqual(result['phase'], 'candidate-import')

    def test_backend_loss_is_blocked(self):
        self.assertEqual(worker.classify_exception(RuntimeError('CUDA device lost'), 'candidate-forward')['status'], 'blocked')
        self.assertEqual(worker.classify_exception(RuntimeError('candidate tensor has wrong shape'), 'candidate-forward')['status'], 'invalid')

    def test_device_assert_is_candidate_invalid_and_storage_runtime_is_error(self):
        self.assertEqual(worker.classify_exception(RuntimeError('CUDA error: device-side assert triggered'), 'candidate-forward')['status'], 'invalid')
        self.assertEqual(worker.classify_exception(RuntimeError('PytorchStreamWriter failed writing file'), 'checkpoint-write')['status'], 'error')
        self.assertEqual(worker.classify_exception(ValueError('bad candidate config'), 'candidate-build')['status'], 'invalid')

    def test_training_deadline_checked_after_last_validation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            candidate = root / 'candidate.py'
            candidate.write_text('''import torch\ndef build_model(): return torch.nn.Linear(1, 1)\ndef build_optimizer(m): return object()\ndef train_step(m, o, x, y, f):\n    with torch.no_grad(): m.weight.add_(1)\n''')
            train = root / 'train.npy'
            np.save(train, np.arange(30, dtype=np.int64))
            args = SimpleNamespace(profile='smoke', seed=0, train=str(train), candidate=str(candidate),
                                   device='cpu', checkpoint=str(root / 'model.pt'))
            protocol = SimpleNamespace(steps=1, context=2, batch=1, train_seconds_limit=10)
            with patch.object(worker, 'get_protocol', return_value=protocol), \
                 patch.object(worker, 'check_model', return_value=1), \
                 patch.object(worker, 'check_causal'), patch.object(worker, 'check_logits'), \
                 patch.object(worker.time, 'monotonic', side_effect=[0, 1, 11]):
                with self.assertRaises(worker.InvalidCandidate):
                    worker.train(args)


if __name__ == '__main__':
    unittest.main()
