"""Bound an entire evaluator subprocess without importing any ML dependencies.

This is process supervision for trusted local code, not a security sandbox.
The controller must load and validate its worker artifacts before writing output.
Separate child sessions are cleaned only after their ancestry has been observed.
A child that detaches and loses its parent between samples may be unidentifiable;
without registration or an OS subreaper it cannot safely be claimed as owned.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


_POLL_SECONDS = 0.2
_PS_TIMEOUT_SECONDS = 0.5
_MAX_CONSECUTIVE_DISCOVERY_TIMEOUTS = 3


def _snapshot(timeout):
    """Return process identities as well as ancestry, with a bounded ps call."""
    result = subprocess.run(
        ['ps', '-axo', 'pid=,ppid=,pgid=,rss=,lstart='],
        capture_output=True, text=True, timeout=timeout, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        pid, ppid, pgid, rss = map(int, fields[:4])
        rows[pid] = {'ppid': ppid, 'pgid': pgid, 'rss': rss * 1024,
                     'identity': fields[4]}
    return rows


class _Tree:
    def __init__(self, root):
        self.root = root
        self.identities = {}
        self.groups = {root}
        self.rows = {}
        self.owned = set()

    def observe(self, rows):
        # A PID/group ID can be reused. A changed birth time invalidates the
        # old association; ancestry may establish fresh ownership below.
        for group in list(self.groups):
            leader = rows.get(group)
            identity = self.identities.get(group)
            if leader is not None and identity is not None and leader['identity'] != identity:
                self.groups.remove(group)
        owned = {pid for pid, identity in self.identities.items()
                 if pid in rows and rows[pid]['identity'] == identity}
        if self.root in rows and self.root not in self.identities:
            owned.add(self.root)
        changed = True
        while changed:
            discovered = {pid for pid, row in rows.items()
                          if row['ppid'] in owned or row['pgid'] in self.groups}
            changed = not discovered.issubset(owned)
            owned.update(discovered)
            for pid in owned:
                row = rows[pid]
                self.identities[pid] = row['identity']
                # Never claim a group led by an unrelated process. A child
                # that joins such a group is cleaned by its individual PID.
                if row['pgid'] == pid:
                    self.groups.add(pid)
        self.rows, self.owned = rows, owned
        return sum(rows[pid]['rss'] for pid in owned)

    def cleanup(self, proc, metadata):
        # Take a final bounded snapshot even after a deadline. This catches
        # children reparented since the last observation and validates PID IDs.
        refreshed = False
        try:
            self.observe(_snapshot(_PS_TIMEOUT_SECONDS))
            refreshed = True
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            metadata['discovery_errors'].append(f'cleanup: {type(exc).__name__}: {exc}')
        groups = {self.root} if proc.poll() is None else set()
        if refreshed:
            groups.update(group for group in self.groups
                          if any(self.rows[pid]['pgid'] == group for pid in self.owned))
        for group in sorted(groups):
            try:
                os.killpg(group, signal.SIGKILL)
                metadata['signaled_groups'].append(group)
            except ProcessLookupError:
                pass
            except OSError as exc:
                metadata['cleanup_errors'].append(f'group {group}: {exc}')
        if refreshed:
            for pid in sorted(self.owned):
                if self.rows[pid]['pgid'] in groups:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    metadata['signaled_pids'].append(pid)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    metadata['cleanup_errors'].append(f'pid {pid}: {exc}')
        try:
            proc.wait(timeout=_PS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            metadata['cleanup_errors'].append('controller did not reap within cleanup budget')


def run(command: list[str], output: Path, timeout_seconds: float,
        rss_limit_bytes: int = 6_000_000_000) -> dict:
    """Run a controller under one monotonic deadline and sampled tree RSS cap.

    stdout/stderr go to ``output.with_suffix('.log')``. Metadata is always
    written to ``output.with_suffix('.supervisor.json')`` and returned on
    success. Exceptions carry that same dict as ``supervisor_metadata``.
    TimeoutError/MemoryError/RuntimeError represent infrastructure failures.
    The supplied output must be a fresh, controller-written result file.
    ``work_wall_s`` covers the original work deadline; ``cleanup_wall_s`` is
    additional bounded process inspection/reaping time after work terminates.
    """
    started = time.monotonic()
    output = Path(output)
    log_path = output.with_suffix('.log')
    metadata_path = output.with_suffix('.supervisor.json')
    metadata = {'command': command, 'output': str(output), 'log': str(log_path),
                'timeout_seconds': timeout_seconds if math.isfinite(timeout_seconds) else None,
                'rss_limit_bytes': rss_limit_bytes if math.isfinite(rss_limit_bytes) else None,
                'status': 'error',
                'pid': None, 'returncode': None, 'peak_sampled_rss_bytes': 0,
                'observed_pids': [], 'signaled_groups': [], 'signaled_pids': [],
                'consecutive_discovery_timeouts': 0,
                'discovery_errors': [], 'cleanup_errors': []}
    proc = tree = None
    failure = None
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise TimeoutError('evaluation wall budget already expired or nonfinite')
        deadline = started + timeout_seconds
        if not math.isfinite(deadline):
            raise TimeoutError('evaluation wall deadline is nonfinite')
        if not math.isfinite(rss_limit_bytes) or rss_limit_bytes <= 0:
            raise MemoryError('process-tree RSS budget must be finite and positive')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RuntimeError('controller output already exists; use a fresh output path')
        with log_path.open('w') as log:
            if time.monotonic() >= deadline:
                raise TimeoutError('evaluation wall budget expired before launch')
            proc = subprocess.Popen(command, stdout=log, stderr=log,
                                    start_new_session=True)
            metadata['pid'] = proc.pid
            tree = _Tree(proc.pid)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('evaluation wall budget exceeded')
                try:
                    rss = tree.observe(_snapshot(min(_PS_TIMEOUT_SECONDS, remaining)))
                except subprocess.TimeoutExpired as exc:
                    metadata['discovery_errors'].append(f'{type(exc).__name__}: {exc}')
                    metadata['consecutive_discovery_timeouts'] += 1
                    if time.monotonic() >= deadline:
                        raise TimeoutError('evaluation wall budget exceeded during process discovery') from exc
                    if metadata['consecutive_discovery_timeouts'] >= _MAX_CONSECUTIVE_DISCOVERY_TIMEOUTS:
                        raise RuntimeError('process discovery timed out 3 consecutive times') from exc
                    # A loaded host may delay a single ps invocation. Retry
                    # within the original deadline; never finish without a
                    # fresh ownership/RSS observation or retry indefinitely.
                    time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
                    continue
                except (subprocess.SubprocessError, OSError, ValueError) as exc:
                    metadata['discovery_errors'].append(f'{type(exc).__name__}: {exc}')
                    raise RuntimeError(f'process discovery failed: {exc}') from exc
                metadata['consecutive_discovery_timeouts'] = 0
                metadata['peak_sampled_rss_bytes'] = max(metadata['peak_sampled_rss_bytes'], rss)
                if time.monotonic() >= deadline:
                    raise TimeoutError('evaluation wall budget exceeded')
                if rss > rss_limit_bytes:
                    raise MemoryError('evaluation process-tree RSS budget exceeded')
                if proc.poll() is not None:
                    break
                time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
            if proc.returncode:
                raise RuntimeError(f'controller exit {proc.returncode}; see {log_path}')
            if not output.is_file():
                raise RuntimeError(f'controller output missing: {output}')
            if time.monotonic() >= deadline:
                raise TimeoutError('evaluation wall budget exceeded checking output')
            metadata['status'] = 'ok'
    except BaseException as exc:
        failure = exc
        metadata['status'] = ('timeout' if isinstance(exc, TimeoutError) else
                              'memory' if isinstance(exc, MemoryError) else 'error')
        metadata['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        cleanup_started = time.monotonic()
        metadata['work_wall_s'] = cleanup_started - started
        if tree is not None:
            tree.cleanup(proc, metadata)
            metadata['observed_pids'] = sorted(tree.identities)
        if proc is not None:
            metadata['returncode'] = proc.returncode
        metadata['cleanup_wall_s'] = time.monotonic() - cleanup_started
        metadata['wall_s'] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    if failure is not None:
        failure.supervisor_metadata = metadata
        raise failure
    if metadata['cleanup_errors']:
        error = RuntimeError('controller cleanup failed; see supervisor metadata')
        error.supervisor_metadata = metadata
        metadata['status'] = 'error'
        metadata['error'] = str(error)
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        raise error
    return metadata
