"""A timeout stops the scorer-owned worker tree, without model/GPU work."""
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scorer import child

class WorkerCleanupTests(unittest.TestCase):
    def test_timeout_cleans_owned_descendant(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            pid_file = directory / 'descendant.pid'
            (directory/'worker.py').write_text(
                'import subprocess,sys,time\nfrom pathlib import Path\n'
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])\n"
                f'Path({str(pid_file)!r}).write_text(str(p.pid))\n'
                'time.sleep(20)\n')
            with self.assertRaises(TimeoutError):
                child(directory/'source.py', directory/'inputs.pt', directory/'output.pt',
                      'cpu', time.monotonic()+1.5)
            self.assertTrue(pid_file.exists(), 'fixture descendant started')
            pid = pid_file.read_text()
            state = subprocess.run(['ps','-o','stat=','-p',pid], text=True, capture_output=True).stdout.strip()
            self.assertTrue(not state or state.startswith('Z'), f'descendant still running: {state}')
            self.assertTrue((directory/'output.worker.json').exists())

if __name__ == '__main__':
    unittest.main()
