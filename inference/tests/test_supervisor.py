"""CPU-only process cleanup with fixture-owned inspection and simulated faults.

Inspection fixtures describe only processes launched by these tests. Detached
controllers acknowledge actual Tree.observe ownership before they can exit, so
test scheduling never depends on a guessed sleep length or global ps latency.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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
        deadline = time.monotonic() + 1
        while True:
            try:
                state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                       text=True, capture_output=True, timeout=0.5).stdout.strip()
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    self.fail(f'could not inspect fixture process {pid} after cleanup')
                continue
            if not state or state.startswith('Z'):
                return
            if time.monotonic() >= deadline:
                self.fail(f'owned descendant {pid} still running: {state}')
            time.sleep(0.01)

    @contextmanager
    def fixture_inspection(self, pid_file=None, ack_file=None):
        """Deterministic known-fixture ancestry, with real OS process groups."""
        original_popen = subprocess.Popen
        original_observe = supervisor._Tree.observe
        controller = None

        def launch(*args, **kwargs):
            nonlocal controller
            controller = original_popen(*args, **kwargs)
            return controller

        def snapshot(timeout):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, supervisor._PS_TIMEOUT_SECONDS)
            processes = [(controller.pid, os.getpid())]
            if pid_file is not None and pid_file.exists():
                processes.append((int(pid_file.read_text()), controller.pid))
            rows = {}
            for pid, ppid in processes:
                try:
                    group = os.getpgid(pid)
                except ProcessLookupError:
                    continue
                rows[pid] = {'ppid': ppid, 'pgid': group, 'rss': 1024,
                             'identity': f'fixture process {pid}'}
            return rows

        def observe(tree, rows):
            rss = original_observe(tree, rows)
            if ack_file is not None and pid_file.exists():
                if int(pid_file.read_text()) in tree.owned:
                    ack_file.touch()
            return rss

        with patch.object(supervisor.subprocess, 'Popen', side_effect=launch), \
                patch.object(supervisor, '_snapshot', side_effect=snapshot), \
                patch.object(supervisor._Tree, 'observe', new=observe):
            yield snapshot

    def detached_child(self):
        pid_file = self.directory / 'descendant.pid'
        ack_file = self.directory / 'observed.ack'
        code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                'fixture_deadline = time.monotonic() + 2.5\n'
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2.5)'], start_new_session=True)\n"
                f'Path({str(pid_file)!r}).write_text(str(child.pid))\n'
                f'while not Path({str(ack_file)!r}).exists():\n'
                '    if time.monotonic() >= fixture_deadline: raise RuntimeError("fixture ownership was not observed")\n'
                '    time.sleep(0.01)\n')
        return pid_file, ack_file, code

    def test_finite_success_and_combined_log(self):
        code = ('import sys\nfrom pathlib import Path\n'
                "print('stdout')\nprint('stderr', file=sys.stderr)\n"
                f'Path({str(self.output)!r}).write_text("{{\\\"ok\\\": true}}")\n')
        with self.fixture_inspection():
            metadata = supervisor.run(self.command(code), self.output, 2)
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
        pid_file, ack_file, code = self.detached_child()
        with self.fixture_inspection(pid_file, ack_file):
            with self.assertRaises(TimeoutError) as caught:
                supervisor.run(self.command(code + 'while time.monotonic() < fixture_deadline: time.sleep(0.01)\n'), self.output, 1.2)
        self.assertTrue(pid_file.is_file(), 'fixture descendant was launched')
        descendant = int(pid_file.read_text())
        self.assertTrue(ack_file.exists(), 'descendant ownership was observed before deadline')
        self.assert_stopped(descendant)
        self.assertIsNone(unrelated.poll(), 'unrelated fixture must remain alive')
        metadata = self.metadata()
        self.assertEqual(metadata['status'], 'timeout')
        self.assertIn(descendant, metadata['observed_pids'])
        self.assertIn(descendant, metadata['signaled_groups'])
        self.assertNotIn(unrelated.pid, metadata['signaled_groups'])
        self.assertEqual(caught.exception.supervisor_metadata, metadata)

    def test_normal_completion_cleans_detached_descendant(self):
        pid_file, ack_file, code = self.detached_child()
        code += f'Path({str(self.output)!r}).write_text("{{}}")\n'
        with self.fixture_inspection(pid_file, ack_file):
            metadata = supervisor.run(self.command(code), self.output, 2)
        self.assertTrue(ack_file.exists())
        self.assertEqual(metadata['status'], 'ok')
        self.assert_stopped(int(pid_file.read_text()))

    def test_nonzero_exit_cleans_detached_descendant(self):
        pid_file, ack_file, code = self.detached_child()
        with self.fixture_inspection(pid_file, ack_file):
            with self.assertRaisesRegex(RuntimeError, 'controller exit 7'):
                supervisor.run(self.command(code + 'sys.exit(7)\n'), self.output, 2)
        self.assertTrue(ack_file.exists())
        self.assert_stopped(int(pid_file.read_text()))
        self.assertEqual(self.metadata()['returncode'], 7)
        self.assertEqual(self.metadata()['status'], 'error')

    def test_missing_output_is_infrastructure_error(self):
        with self.fixture_inspection():
            with self.assertRaisesRegex(RuntimeError, 'controller output missing'):
                supervisor.run(self.command('pass'), self.output, 2)
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
        with self.fixture_inspection():
            with self.assertRaises(MemoryError):
                supervisor.run(self.command('import time; time.sleep(2.5)'), self.output, 2,
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
        with self.simulated_supervision(0.06) as state:
            with self.assertRaises(TimeoutError):
                supervisor.run(self.command('fixture controller'), self.output, 0.06)
        self.assertAlmostEqual(state['timeouts'][0], 0.06)
        self.assertAlmostEqual(self.metadata()['work_wall_s'], 0.06)
        self.assertEqual(self.metadata()['status'], 'timeout')

    @contextmanager
    def simulated_supervision(self, first_timeout_cost):
        """Virtual clock proves work deadlines independently of cleanup slack."""
        state = {'time': 100.0, 'timeouts': [], 'cleanup_started': False}
        process = Mock(pid=987654, returncode=None)
        process.poll.return_value = None

        def sleep(seconds):
            state['time'] += seconds

        def snapshot(timeout):
            state['timeouts'].append(timeout)
            if len(state['timeouts']) == 1:
                sleep(first_timeout_cost)
                raise subprocess.TimeoutExpired(['ps'], timeout)
            return {process.pid: {'ppid': os.getpid(), 'pgid': process.pid,
                                  'rss': 1024, 'identity': 'virtual fixture'}}

        original_cleanup = supervisor._Tree.cleanup

        def cleanup(tree, proc, metadata):
            state['cleanup_started'] = True
            # Simulate a bounded 0.4 s ps inspection and 0.3 s reap, which
            # are deliberately outside the original child work budget.
            sleep(0.4)
            original_cleanup(tree, proc, metadata)

        def wait(timeout):
            sleep(0.3)
            process.returncode = -9
            return -9

        process.wait.side_effect = wait
        with patch.object(supervisor.time, 'monotonic', side_effect=lambda: state['time']), \
                patch.object(supervisor.time, 'sleep', side_effect=sleep), \
                patch.object(supervisor.subprocess, 'Popen', return_value=process), \
                patch.object(supervisor, '_snapshot', side_effect=snapshot), \
                patch.object(supervisor._Tree, 'cleanup', new=cleanup), \
                patch.object(supervisor.os, 'killpg') as kill_group:
            yield state
            kill_group.assert_called_once_with(process.pid, supervisor.signal.SIGKILL)

    def test_transient_ps_timeout_recovers_and_records_failure(self):
        calls = 0
        code = ('import time\nfrom pathlib import Path\n'
                f'time.sleep(0.1)\nPath({str(self.output)!r}).write_text("{{}}")\n')
        with self.fixture_inspection() as original_snapshot:
            def intermittent_snapshot(timeout):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise subprocess.TimeoutExpired(['ps'], timeout)
                return original_snapshot(timeout)

            with patch.object(supervisor, '_snapshot', side_effect=intermittent_snapshot):
                metadata = supervisor.run(self.command(code), self.output, 2)
        self.assertEqual(metadata['status'], 'ok')
        self.assertEqual(metadata['consecutive_discovery_timeouts'], 0)
        self.assertEqual(len(metadata['discovery_errors']), 1)
        self.assertEqual(metadata, self.metadata())

    def test_three_consecutive_ps_timeouts_are_infrastructure_error(self):
        calls = 0
        with self.fixture_inspection() as original_snapshot:
            def stalled_snapshot(timeout):
                nonlocal calls
                calls += 1
                if calls <= 3:
                    raise subprocess.TimeoutExpired(['ps'], timeout)
                return original_snapshot(timeout)

            with patch.object(supervisor, '_snapshot', side_effect=stalled_snapshot):
                with self.assertRaisesRegex(RuntimeError, '3 consecutive times') as caught:
                    supervisor.run(self.command('import time; time.sleep(2.5)'), self.output, 2)
        metadata = self.metadata()
        self.assertEqual(metadata['status'], 'error')
        self.assertEqual(metadata['consecutive_discovery_timeouts'], 3)
        self.assertEqual(len(metadata['discovery_errors']), 3)
        self.assertEqual(calls, 4, 'three monitoring attempts plus final ownership cleanup')
        self.assertEqual(caught.exception.supervisor_metadata, metadata)
        self.assert_stopped(metadata['pid'])

    def test_transient_ps_timeout_does_not_extend_original_deadline(self):
        with self.simulated_supervision(0.04) as state:
            with self.assertRaises(TimeoutError):
                supervisor.run(self.command('fixture controller'), self.output, 0.32)
        metadata = self.metadata()
        self.assertTrue(state['cleanup_started'])
        self.assertAlmostEqual(metadata['work_wall_s'], 0.32)
        self.assertAlmostEqual(metadata['cleanup_wall_s'], 0.7)
        self.assertAlmostEqual(metadata['wall_s'], 1.02)
        self.assertEqual(metadata['status'], 'timeout')
        self.assertEqual(metadata['timeout_seconds'], 0.32)
        self.assertEqual(metadata['consecutive_discovery_timeouts'], 0,
                         'successful observation reset retry counter before deadline')
        self.assertEqual(len(metadata['discovery_errors']), 1)

    def test_ps_snapshot_parses_ancestry_birth_identity_and_rss(self):
        result = subprocess.CompletedProcess([], 0,
            stdout='100 1 100 20 Wed Oct 14 10:00:00 2026\n101 100 101 30 Wed Oct 14 10:00:01 2026\n',
            stderr='')
        with patch.object(supervisor.subprocess, 'run', return_value=result) as inspect:
            rows = supervisor._snapshot(0.125)
        self.assertEqual(rows[101]['ppid'], 100)
        self.assertEqual(rows[101]['pgid'], 101)
        self.assertEqual(rows[101]['rss'], 30 * 1024)
        self.assertEqual(rows[101]['identity'], 'Wed Oct 14 10:00:01 2026')
        self.assertEqual(inspect.call_args.kwargs['timeout'], 0.125)
        self.assertTrue(inspect.call_args.kwargs['check'])

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
