"""Distinguish bad policy code from infrastructure, without model execution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorer

class ExecutionStatusTests(unittest.TestCase):
    def run_failure(self, failure_index):
        calls = []
        def worker(*args):
            calls.append(args)
            if len(calls) == failure_index:
                raise RuntimeError('fixture worker exited with exception')
            return {'conditions': []}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(scorer, 'local_model'), \
             patch.object(scorer, 'machine', return_value={'blockers': []}), \
             patch.object(scorer, 'make_inputs', return_value={'task_id':'fixture-train','decode_steps':8,'cases':[]}), \
             patch.object(scorer, 'child', side_effect=worker):
            return scorer._evaluate_with_errors(scorer.HERE/'solution.py', Path(root)/'episode')

    def test_candidate_exception_is_invalid_zero(self):
        record = self.run_failure(2)['record']
        self.assertEqual(record['status'], 'invalid')
        self.assertEqual(record['reward'], 0)

    def test_reference_exception_is_infrastructure_null(self):
        record = self.run_failure(1)['record']
        self.assertEqual(record['status'], 'error')
        self.assertIsNone(record['reward'])

if __name__ == '__main__':
    unittest.main()
