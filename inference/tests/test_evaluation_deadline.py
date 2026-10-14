"""Actual hung evaluator preflight is excluded; no candidate/model is run."""
import sys
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scorer

class EvaluationDeadlineTests(unittest.TestCase):
    def test_hung_asset_validation_returns_excluded_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            marker=root/'entered-preflight'
            script=root/'hung-controller.py'
            script.write_text('import sys,time\nfrom pathlib import Path\n'
                f'sys.path.insert(0,{str(scorer.HERE)!r})\n'
                'import scorer\nimport controller\n'
                f'def hung():\n Path({str(marker)!r}).write_text("entered")\n time.sleep(30)\n'
                'scorer.local_model=hung\n'
                'controller.main(*sys.argv[1:])\n')
            with patch.dict(os.environ,RSI_INFERENCE_MEASUREMENT_LOCK=str(root/'measurement.lock')), \
                 patch.object(scorer,'controller_command',side_effect=lambda c,o:[sys.executable,str(script),str(c),str(o)]), \
                 patch.dict(scorer.LIMITS,evaluation_wall_s=10):
                start=time.monotonic()
                result=scorer.evaluate(scorer.HERE/'solution.py',root/'episode')
                elapsed=time.monotonic()-start
            self.assertTrue(marker.exists(),'hung preflight actually entered')
            self.assertLess(elapsed,13)
            self.assertEqual(result['record']['status'],'error')
            self.assertIsNone(result['reward'])
            self.assertIsNone(result['record']['reward'])
            self.assertIn('TimeoutError',result['error'])
            self.assertFalse(result['record']['metrics']['correctness_pass'])
            self.assertNotIn('candidate',result.get('workers',{}))

if __name__ == '__main__':
    unittest.main()
