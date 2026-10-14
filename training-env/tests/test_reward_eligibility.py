"""Native declarations protect smoke and heldout feedback from policy updates."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from interaction import TrainingEnv

class EligibilityTests(unittest.TestCase):
    def test_native_eligibility_and_reopen_feedback(self):
        for profile,split,seed,expected in [('standard','train',0,True),('standard','heldout',10000,False),('smoke','train',0,False)]:
            with self.subTest(profile=profile,split=split), tempfile.TemporaryDirectory() as directory:
                kwargs=dict(profile=profile,task_split=split,seed=seed)
                env=TrainingEnv(Path(directory)/'episode',directory,**kwargs)
                record={'status':'ok','metrics':{'held_out_loss':8.0},'reward':0.1}
                with patch('interaction.evaluate',return_value=record):
                    intermediate=env.step({'code':'# candidate'})
                    self.assertFalse(intermediate['info']['trainable'])
                    result=env.step({'code':'# final','final':True})
                self.assertEqual(result['info']['trainable'],expected)
                reopened=TrainingEnv(Path(directory)/'episode',directory,**kwargs)
                self.assertEqual(reopened.reset()['last_feedback'],result['observation'])
                self.assertEqual(result['reward'],0.1) # eligibility never rewrites measured reward

    def test_invalid_standard_terminal_trainable_but_infra_excluded(self):
        for status,expected in [('invalid',True),('blocked',False),('error',False)]:
            with tempfile.TemporaryDirectory() as directory:
                env=TrainingEnv(Path(directory)/'episode',directory,profile='standard')
                record={'status':status,'metrics':{},'reward':0 if status=='invalid' else None}
                with patch('interaction.evaluate',return_value=record):
                    result=env.step({'code':'# candidate','final':True})
                self.assertEqual(result['info']['trainable'],expected)

if __name__=='__main__':unittest.main()
