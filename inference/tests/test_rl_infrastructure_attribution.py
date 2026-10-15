"""Remote CPU only: mocked evaluator; no model/asset execution or timing reward."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_environment


class RLInfrastructureAttributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='inference-api-errors-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluation = patch.object(rl_environment, 'evaluate')
        self.evaluate = self.evaluation.start()
        self.addCleanup(self.evaluation.stop)
        self.counter = 0

    def environment(self, **kwargs):
        self.counter += 1
        value = rl_environment.CodingAgentEnvironment(self.root / str(self.counter), **kwargs)
        observation = value.reset(split='train', seed=7)
        return value, observation['starter_code']

    def native_record(self, environment, status, reward):
        return rl_environment.make_record(environment='inference', task=environment.spec['task_id'],
            profile='standard', status=status, reward=reward,
            metrics={'correctness_pass':status == 'ok', 'timing_valid':status == 'ok'},
            provenance={'split':'train', 'synthetic':True})

    def test_invalid_device_configuration_fails_before_episode_or_action(self):
        for device in ('cdua', '', None, True, 42, {}):
            with self.subTest(device=device), self.assertRaises(ValueError):
                rl_environment.CodingAgentEnvironment(self.root / 'not-created', device=device)
        self.assertFalse((self.root / 'not-created').exists())
        self.evaluate.assert_not_called()

    def test_nonboolean_diagnostic_configuration_fails_before_episode(self):
        for diagnostic in ('false', 'true', None, 0, 1):
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ValueError):
                rl_environment.CodingAgentEnvironment(self.root / 'not-created', diagnostic=diagnostic)
        self.assertFalse((self.root / 'not-created').exists())
        self.evaluate.assert_not_called()

    def test_supported_devices_reset_without_evaluator_or_availability_probe(self):
        # Availability remains an evaluator concern; these CPU fixture resets
        # must not initialize CUDA or load assets for a selected device.
        import torch
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('unexpected probe')):
            for device in ('cpu', 'mps', 'cuda'):
                with self.subTest(device=device):
                    environment, _ = self.environment(device=device)
                    self.assertEqual(environment.device, device)
                    self.assertFalse(environment.done)
        self.evaluate.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_escaping_evaluator_errors_terminate_with_null_excluded_reward(self):
        for error_type in (ValueError, TypeError, SyntaxError, UnicodeError, RuntimeError):
            with self.subTest(error_type=error_type.__name__):
                environment, code = self.environment(max_steps=3)
                self.evaluate.side_effect = error_type('synthetic evaluator failure')
                transition = environment.step({'type':'submit', 'code':code})
                self.assertTrue(transition['terminated'])
                self.assertIsNone(transition['reward'])
                self.assertIsNone(transition['info']['terminal_reward'])
                self.assertFalse(transition['info']['trainable'])
                record = transition['info']['record']
                self.assertEqual(record['status'], 'error')
                self.assertIsNone(record['reward'])
                self.assertIsNone(record['metrics']['model_executed'])
                self.assertEqual(record['provenance']['stage'], 'evaluation')
                self.assertEqual(environment.used, 1)
                with self.assertRaises(ValueError):
                    environment.step({'type':'submit', 'code':code})

    def test_invalid_action_or_syntax_retains_zero_without_evaluator(self):
        for action in ({'type':'submit', 'code':'class Engine(:'},
                       {'type':'submit', 'code':'pass', 'finish':'true'}):
            with self.subTest(action=action):
                environment, _ = self.environment(max_steps=1)
                transition = environment.step(action)
                self.assertTrue(transition['terminated'])
                self.assertEqual(transition['reward'], 0.0)
                self.assertTrue(transition['info']['trainable'])
                self.assertEqual(transition['info']['record']['status'], 'invalid')
                self.assertEqual(transition['info']['record']['provenance']['stage'], 'action_validation')
        self.evaluate.assert_not_called()

    def test_reported_native_statuses_and_rewards_are_preserved(self):
        for status, reward in (('ok', .51), ('invalid', 0.0), ('blocked', None), ('error', None)):
            with self.subTest(status=status):
                environment, code = self.environment(max_steps=1)
                record = self.native_record(environment, status, reward)
                self.evaluate.return_value = {'record':record}
                transition = environment.step({'type':'submit', 'code':code})
                self.assertEqual(transition['info']['record'], record)
                self.assertEqual(transition['reward'], reward)
                self.assertEqual(transition['info']['terminal_reward'], reward)
                self.assertEqual(transition['info']['trainable'], status in ('ok', 'invalid'))

    def test_nonterminal_valid_feedback_keeps_zero_intermediate_reward(self):
        environment, code = self.environment(max_steps=3)
        self.evaluate.return_value = {'record':self.native_record(environment, 'ok', .51)}
        transition = environment.step({'type':'submit', 'code':code})
        self.assertFalse(transition['terminated'])
        self.assertEqual(transition['reward'], 0.0)
        self.assertIsNone(transition['info']['terminal_reward'])
        self.assertEqual(transition['info']['record']['reward'], .51)


if __name__ == '__main__':
    unittest.main()
