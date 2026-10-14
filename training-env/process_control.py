"""Bounded cleanup of observed reproduction-owned subprocesses; not a sandbox.

Process identity/ancestry core reused from sibling inference/supervisor.py.
Children detached before observation cannot be safely claimed as owned.
"""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import signal

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


class CommandTimeout(TimeoutError):
    pass

def run_command(command, *, cwd, env, timeout=150, max_output_bytes=8_000_000):
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('command timeout must be finite and positive')
    started=time.monotonic();deadline=started+timeout
    metadata={'status':'running','pid':None,'observed_pids':[], 'signaled_groups':[], 'signaled_pids':[],
              'discovery_errors':[], 'cleanup_errors':[], 'timeout_seconds':timeout}
    process=tree=None;failure=None
    with tempfile.TemporaryFile(dir=cwd) as stdout,tempfile.TemporaryFile(dir=cwd) as stderr:
        try:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            metadata['pid']=process.pid;tree=_Tree(process.pid)
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise CommandTimeout('reproduction command wall timeout')
                try:tree.observe(_snapshot(min(_PS_TIMEOUT_SECONDS,remaining)))
                except subprocess.TimeoutExpired as exc:
                    metadata['discovery_errors'].append(str(exc))
                    if time.monotonic()>=deadline:
                        raise CommandTimeout('reproduction deadline expired during process discovery') from exc
                    raise RuntimeError('process discovery failed within reproduction budget') from exc
                except (subprocess.SubprocessError,OSError,ValueError) as exc:
                    metadata['discovery_errors'].append(str(exc))
                    raise RuntimeError('cannot safely supervise reproduction-owned processes') from exc
                if os.fstat(stdout.fileno()).st_size+os.fstat(stderr.fileno()).st_size>max_output_bytes:
                    raise RuntimeError('reproduction output exceeds capture limit')
                if process.poll() is not None:break
                time.sleep(min(.05,max(0,deadline-time.monotonic())))
            metadata['status']='ok' if process.returncode==0 else 'error'
        except BaseException as exc:
            failure=exc;metadata['status']='timeout' if isinstance(exc,CommandTimeout) else 'error'
        finally:
            cleanup_started=time.monotonic()
            if tree is not None:
                tree.cleanup(process,metadata);metadata['observed_pids']=sorted(tree.identities)
            metadata['work_seconds']=cleanup_started-started
            metadata['cleanup_seconds']=time.monotonic()-cleanup_started
            metadata['wall_seconds']=time.monotonic()-started
        stdout.seek(0);stderr.seek(0)
        out=stdout.read(max_output_bytes).decode('utf-8',errors='replace')
        err=stderr.read(max_output_bytes).decode('utf-8',errors='replace')
    if failure is not None:
        failure.supervisor_metadata=metadata;raise failure
    if metadata['cleanup_errors']:
        failure=RuntimeError('reproduction process cleanup incomplete')
        failure.supervisor_metadata=metadata;raise failure
    result=subprocess.CompletedProcess(command,process.returncode,out,err)
    result.supervisor_metadata=metadata
    return result
