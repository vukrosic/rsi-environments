import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from interaction import TrainingEnv

class SessionPersistenceTests(unittest.TestCase):
    def test_corrupt_budget_never_grants_more_attempts(self):
        for changes in ({'trials':-1},{'trials':11},{'trials':True},{'finalized':0},{'config':{}}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as directory:
                env=TrainingEnv(Path(directory)/'session',directory)
                state=json.loads(env.state_path.read_text());state.update(changes)
                env.state_path.write_text(json.dumps(state))
                with patch('interaction.evaluate') as scorer:
                    with self.assertRaises(ValueError):env.step({'code':'# source'})
                scorer.assert_not_called()

    def test_terminal_crash_preserves_seal_after_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session';env=TrainingEnv(path,directory)
            with patch('interaction.evaluate',side_effect=OSError('operator I/O interruption')):
                with self.assertRaises(OSError):env.step({'code':'# source','final':True})
            reopened=TrainingEnv(path,directory)
            self.assertTrue(reopened.reset()['finalized'])
            with self.assertRaises(RuntimeError):reopened.step({'code':'# replay','final':True})

    def test_successful_action_feedback_survives_new_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session';env=TrainingEnv(path,directory,max_trials=1)
            with patch('interaction.evaluate',return_value={'status':'invalid','reward':0.,'metrics':{'reason':'fixture'}}):
                result=env.step({'code':'# source'})
            reopened=TrainingEnv(path,directory,max_trials=1)
            self.assertEqual(reopened.reset()['last_feedback'],result['observation'])
            self.assertEqual(reopened.reset()['trials_remaining'],0)
            with self.assertRaises(RuntimeError):reopened.step({'code':'# again'})

if __name__=='__main__':unittest.main()
