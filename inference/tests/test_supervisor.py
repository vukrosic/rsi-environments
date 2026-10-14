"""CPU-only supervision checks; no torch/model/accelerator imports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import supervisor


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / 'output.json'

    def command(self, code):
        return [sys.executable, '-c', code]

    def metadata(self):
        return json.loads(self.output.with_suffix('.supervisor.json').read_text())

    def assert_stopped(self, pid):
        # An orphan can remain a zombie until its adopter reaps it.
        deadline = time.monotonic() + 0.3
        while True:
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                   text=True, capture_output=True, timeout=0.2).stdout.strip()
            if not state or state.startswith('Z'):
                return
            if time.monotonic() >= deadline:
                self.fail(f'owned descendant {pid} still running: {state}')
            time.sleep(0.01)

    def detached_child(self):
        pid_file = self.directory / 'descendant.pid'
        code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2.5)'], start_new_session=True)\n"
                f'Path({str(pid_file)!r}).write_text(str(child.pid))\n')
        return pid_file, code

    def test_finite_success_and_combined_log(self):
        code = ('import sys\nfrom pathlib import Path\n'
                "print('stdout')\nprint('stderr', file=sys.stderr)\n"
                f'Path({str(self.output)!r}).write_text("{{\\\"ok\\\": true}}")\n')
        metadata = supervisor.run(self.command(code), self.output, 1)
        self.assertEqual(metadata['status'], 'ok')
        self.assertEqual(metadata['returncode'], 0)
        self.assertTrue(json.loads(self.output.read_text())['ok'])
        self.assertEqual(metadata, self.metadata())
        self.assertIn('stdout', self.output.with_suffix('.log').read_text())
        self.assertIn('stderr', self.output.with_suffix('.log').read_text())

    def test_timeout_cleans_detached_descendant_and_preserves_unrelated(self):
        unrelated = subprocess.Popen(self.command('import time; time.sleep(2.5)'),
                                     start_new_session=True)
        def stop_unrelated_fixture():
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=0.3)
        self.addCleanup(stop_unrelated_fixture)
        pid_file, code = self.detached_child()
        start = time.monotonic()
        with self.assertRaises(TimeoutError) as caught:
            supervisor.run(self.command(code + 'time.sleep(2.5)\n'), self.output, 0.45)
        self.assertLess(time.monotonic() - start, 1.1)
        self.assertTrue(pid_file.is_file(), 'fixture descendant was launched')
        descendant = int(pid_file.read_text())
        self.assert_stopped(descendant)
        self.assertIsNone(unrelated.poll(), 'unrelated fixture must remain alive')
        metadata = self.metadata()
        self.assertEqual(metadata['status'], 'timeout')
        self.assertIn(descendant, metadata['observed_pids'])
        self.assertIn(descendant, metadata['signaled_groups'])
        self.assertNotIn(unrelated.pid, metadata['signaled_groups'])
        self.assertEqual(caught.exception.supervisor_metadata, metadata)

    def test_normal_completion_cleans_detached_descendant(self):
        pid_file, code = self.detached_child()
        code += f'time.sleep(0.2)\nPath({str(self.output)!r}).write_text("{{}}")\n'
        metadata = supervisor.run(self.command(code), self.output, 1)
        self.assertEqual(metadata['status'], 'ok')
        self.assert_stopped(int(pid_file.read_text()))

    def test_nonzero_exit_cleans_detached_descendant(self):
        pid_file, code = self.detached_child()
        with self.assertRaisesRegex(RuntimeError, 'controller exit 7'):
            supervisor.run(self.command(code + 'time.sleep(0.2)\nsys.exit(7)\n'), self.output, 1)
        self.assert_stopped(int(pid_file.read_text()))
        self.assertEqual(self.metadata()['returncode'], 7)
        self.assertEqual(self.metadata()['status'], 'error')

    def test_missing_output_is_infrastructure_error(self):
        with self.assertRaisesRegex(RuntimeError, 'controller output missing'):
            supervisor.run(self.command('pass'), self.output, 1)
        self.assertEqual(self.metadata()['status'], 'error')

    def test_invalid_or_expired_timeout_never_launches(self):
        for budget in (0, -1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(budget=budget), patch.object(supervisor.subprocess, 'Popen') as launch:
                with self.assertRaises(TimeoutError):
                    supervisor.run(self.command('pass'), self.output, budget)
                launch.assert_not_called()
                self.assertEqual(self.metadata()['status'], 'timeout')
                self.assertIsNone(self.metadata()['pid'])

    def test_memory_cap_is_tree_sampled_and_durable(self):
        with self.assertRaises(MemoryError):
            supervisor.run(self.command('import time; time.sleep(2.5)'), self.output, 1,
                           rss_limit_bytes=1)
        metadata = self.metadata()
        self.assertEqual(metadata['status'], 'memory')
        self.assertGreater(metadata['peak_sampled_rss_bytes'], 1)
        self.assert_stopped(metadata['pid'])

    def test_rejects_stale_output_without_launch(self):
        self.output.write_text('{}')
        with patch.object(supervisor.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                supervisor.run(self.command('pass'), self.output, 1)
            launch.assert_not_called()
        self.assertEqual(self.metadata()['status'], 'error')

    def test_ps_sampling_is_bounded_by_remaining_deadline(self):
        original_snapshot = supervisor._snapshot
        timeouts = []

        def stalled_snapshot(timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                time.sleep(timeout)
                raise subprocess.TimeoutExpired(['ps'], timeout)
            return original_snapshot(timeout)

        with patch.object(supervisor, '_snapshot', side_effect=stalled_snapshot):
            with self.assertRaises(TimeoutError):
                supervisor.run(self.command('import time; time.sleep(2.5)'), self.output, 0.06)
        self.assertLessEqual(timeouts[0], 0.06)
        self.assertEqual(self.metadata()['status'], 'timeout')
        self.assert_stopped(self.metadata()['pid'])

    def test_tree_rss_tracks_orphans_without_claiming_unrelated_group(self):
        def row(ppid, pgid, rss, identity):
            return {'ppid': ppid, 'pgid': pgid, 'rss': rss, 'identity': identity}
        tree = supervisor._Tree(100)
        rows = {100: row(1, 100, 10, 'controller'),
                101: row(100, 101, 20, 'detached child'),
                102: row(101, 101, 30, 'grandchild'),
                200: row(1, 200, 1000, 'unrelated')}
        self.assertEqual(tree.observe(rows), 60)
        self.assertEqual(tree.groups, {100, 101})
        rows.pop(100)
        rows[101]['ppid'] = 1
        self.assertEqual(tree.observe(rows), 50)
        self.assertNotIn(200, tree.owned)
        # A reused group leader PID must lose its old ownership.
        rows[101] = row(1, 101, 2000, 'reused unrelated PID')
        rows.pop(102)
        self.assertEqual(tree.observe(rows), 0)
        self.assertNotIn(101, tree.groups)


if __name__ == '__main__':
    unittest.main()
