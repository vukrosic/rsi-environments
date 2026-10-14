"""Reset cannot move an active source action into a different policy episode."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rl_environment
from rl_environment import CodingAgentEnvironment

class ResetSerializationTests(unittest.TestCase):
    def test_reset_waits_for_step_and_preserves_old_episode_receipt(self):
        entered=threading.Event();release=threading.Event();reset_attempted=threading.Event()
        completed=threading.Event();results={};errors=[]
        def evaluate(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise RuntimeError('fixture timed out')
            return {'record':{'status':'ok','reward':.5}}
        with tempfile.TemporaryDirectory() as root:
            env=CodingAgentEnvironment(root,max_steps=1)
            observation=env.reset(split='train',seed=1);original=env.episode
            def step():
                try:results['step']=env.step({'type':'submit','code':observation['starter_code']})
                except BaseException as exc:errors.append(exc)
            def reset():
                reset_attempted.set()
                try:results['reset']=env.reset(split='heldout',seed=2)
                except BaseException as exc:errors.append(exc)
                finally:completed.set()
            with patch.object(rl_environment,'evaluate',side_effect=evaluate):
                first=threading.Thread(target=step);second=threading.Thread(target=reset)
                first.start();self.assertTrue(entered.wait(3))
                second.start();self.assertTrue(reset_attempted.wait(3))
                try:self.assertFalse(completed.wait(.1),'reset crossed an active step')
                finally:release.set();first.join(5);second.join(5)
            self.assertFalse(errors,errors)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(results['step']['info']['episode_dir'],str(original))
            self.assertEqual(results['step']['info']['split'],'train')
            self.assertTrue((original/'transitions.jsonl').is_file())
            self.assertEqual(results['reset']['split'],'heldout')
            self.assertNotEqual(env.episode,original)

if __name__=='__main__':unittest.main()
