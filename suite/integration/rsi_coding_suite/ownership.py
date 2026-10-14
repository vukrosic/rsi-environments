"""Identity-bound cleanup of registered/observed owned evaluator processes."""
import asyncio
import json
import os
from pathlib import Path
import signal
import uuid
import math

import psutil


async def finish_cleanup(task):
    """Propagate caller cancellation only after the retained teardown completes.

    Shield alone permits the caller to return before teardown. This loop also
    handles repeated cancellation while preserving one shared cleanup task.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class OwnedProcesses:
    def __init__(self, folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.ledger=self.folder/'owned-processes.jsonl';self.token=uuid.uuid4().hex
        handle=os.open(self.ledger,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(handle)
        self.entries={}
        self._cleanup_task=None

    def environment(self):
        env=dict(os.environ)
        env['RSI_OWNED_LEDGER']=str(self.ledger.resolve());env['RSI_OWNED_TOKEN']=self.token
        env['PYTHONPATH']=str(Path(__file__).with_name('ownership_site'))+os.pathsep+env.get('PYTHONPATH','')
        return env

    def register(self,pid):
        try:self.entries[pid]=psutil.Process(pid).create_time()
        except psutil.Error:pass

    def refresh(self):
        for line in self.ledger.read_text().splitlines():
            try:entry=json.loads(line)
            except ValueError:continue  # concurrent final append may be incomplete
            if (isinstance(entry,dict) and entry.get('token')==self.token
                    and type(entry.get('pid')) is int and entry['pid']>0
                    and type(entry.get('created')) in (int,float)
                    and math.isfinite(entry['created'])):
                self.entries[entry['pid']]=entry['created']
        for pid,created in list(self.entries.items()):
            try:
                parent=psutil.Process(pid)
                if parent.create_time()!=created:continue
                for child in parent.children(recursive=True):self.entries[child.pid]=child.create_time()
            except psutil.Error:pass
        live=[]
        for pid,created in self.entries.items():
            try:
                process=psutil.Process(pid)
                if process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:live.append(pid)
            except psutil.Error:pass
        return live

    def signal(self,pid,kind):
        try:
            if psutil.Process(pid).create_time()!=self.entries.get(pid):return False
            os.kill(pid,kind);return True
        except (ProcessLookupError,psutil.Error):return False

    async def cleanup(self):
        if self._cleanup_task is None:
            self._cleanup_task=asyncio.create_task(self._cleanup())
        return await finish_cleanup(self._cleanup_task)

    async def _cleanup(self):
        signaled=[]
        # Freeze identity-checked live owners before discovery/reaping so a
        # cooperative parent cannot launch more workers during teardown.
        try:
            for _ in range(3):
                live=self.refresh()
                for pid in live:
                    self.signal(pid,signal.SIGSTOP)
                await asyncio.sleep(.02)
        finally:
            # Once an owner has been stopped, always attempt its terminal signal.
            # Caller cancellation cannot interrupt this dedicated cleanup task.
            for pid in self.refresh():
                if self.signal(pid,signal.SIGKILL):signaled.append(pid)
        deadline=asyncio.get_running_loop().time()+1
        while self.refresh() and asyncio.get_running_loop().time()<deadline:await asyncio.sleep(.02)
        remaining=self.refresh()
        result={'signaled_owned_pids':signaled,'remaining_owned_pids':remaining,
                'registered_pid_count':len(self.entries),'identity':'PID plus process creation time'}
        (self.folder/'process-cleanup.json').write_text(json.dumps(result,indent=2))
        if remaining:raise RuntimeError(f'owned processes not reaped: {remaining}')
        return result
