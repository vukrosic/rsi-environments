import fcntl
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interaction import SessionBusy, TrainingEnv
from suite.records import validate_record


class SessionWaitTests(unittest.TestCase):
    def test_busy_requests_are_bounded_and_leave_owner_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            env = TrainingEnv(Path(directory) / 'session', directory, lock_timeout_seconds=.03)
            before = env.state_path.read_bytes()
            with (env.root / '.session.lock').open('a+') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('interaction.evaluate') as scorer:
                    results = []
                    for submit in (lambda: env.step({'code': '# code'}),
                                   lambda: env.step({'code': '# code', 'final': True}),
                                   lambda: env.final('missing.py')):
                        start = time.monotonic()
                        result = submit()
                        self.assertLess(time.monotonic() - start, .5)
                        self.assertTrue(result['truncated'])
                        self.assertIsNone(result['reward'])
                        self.assertEqual(result['observation']['status'], 'blocked')
                        self.assertFalse(result['info']['trainable'])
                        self.assertTrue(result['info']['exclude_from_rl'])
                        self.assertTrue(result['info']['session_state_unchanged'])
                        self.assertEqual(result['info']['operational_phase'], 'lockbusy')
                        self.assertEqual(validate_record(result['record']), result['record'])
                        self.assertEqual(json.loads(Path(result['info']['record_path']).read_text()), result['record'])
                        self.assertEqual(env.state_path.read_bytes(), before)
                        results.append(result)
                    self.assertEqual(len({r['info']['record_path'] for r in results}), 3)
                    scorer.assert_not_called()
                for operation in (env.reset, lambda: TrainingEnv(env.root, directory, lock_timeout_seconds=.03)):
                    start = time.monotonic()
                    with self.assertRaisesRegex(SessionBusy, 'blocked infrastructure'):
                        operation()
                    self.assertLess(time.monotonic() - start, .5)
                self.assertEqual(env.state_path.read_bytes(), before)
            self.assertFalse(env.reset()['finalized'])
            self.assertEqual(env.reset()['trials_remaining'], 10)

    def test_timeout_is_operational_and_old_sessions_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            env = TrainingEnv(Path(directory) / 'session', directory)
            before = env.state_path.read_bytes()
            self.assertNotIn('lock_timeout_seconds', json.loads(before)['config'])
            reopened = TrainingEnv(env.root, directory, lock_timeout_seconds=0)
            self.assertEqual(env.config, reopened.config)
            self.assertEqual(before, reopened.state_path.read_bytes())
            with patch('interaction.evaluate', return_value={'status': 'invalid', 'reward': 0., 'metrics': {'reason': 'fixture'}}):
                result = reopened.step({'code': '# source'})
            self.assertFalse(result['truncated'])
            self.assertEqual(reopened.reset()['trials_remaining'], 9)

    def test_invalid_timeouts(self):
        with tempfile.TemporaryDirectory() as directory:
            for timeout in (True, False, float('nan'), float('inf'), -.01, 60.01, '2', None):
                with self.subTest(timeout=timeout):
                    with self.assertRaisesRegex(ValueError, 'lock_timeout_seconds'):
                        TrainingEnv(Path(directory) / 'session', directory, lock_timeout_seconds=timeout)


if __name__ == '__main__':
    unittest.main()
