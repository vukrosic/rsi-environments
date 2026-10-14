import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from process_control import run_command,CommandTimeout

class ProcessControlTests(unittest.TestCase):
    def test_success_preserves_output_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run_command([sys.executable,'-c','print("ready")'],cwd=directory,env=os.environ.copy(),timeout=2)
            self.assertEqual(result.stdout,'ready\n');self.assertEqual(result.returncode,0)
            self.assertEqual(result.supervisor_metadata['status'],'ok')

    def test_timeout_cleans_owned_child_and_keeps_unrelated_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);child_pid=root/'child.pid'
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            try:
                code='import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)'
                with self.assertRaises(CommandTimeout) as caught:
                    run_command([sys.executable,'-c',code,str(child_pid)],cwd=directory,env=os.environ.copy(),timeout=.6)
                metadata=caught.exception.supervisor_metadata
                pid=int(child_pid.read_text());self.assertIn(pid,metadata['observed_pids'])
                for _ in range(20):
                    state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
                    if not state or state.startswith('Z'):break
                    time.sleep(.02)
                self.assertTrue(not state or state.startswith('Z'),state)
                self.assertIsNone(unrelated.poll())
                self.assertLess(metadata['wall_seconds'],2.5)
            finally:
                unrelated.kill();unrelated.wait()

    def test_invalid_timeout_fails_before_spawn(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            with patch('process_control.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):run_command(['unused'],cwd='.',env={},timeout=value)
                spawn.assert_not_called()

if __name__=='__main__':unittest.main()
