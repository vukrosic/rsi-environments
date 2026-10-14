"""Native bridge deadlines/ordinary owned processes; tiny CPU fixtures only."""
import asyncio
import json
from pathlib import Path
import sys
import time

import psutil
import pytest
from rsi_coding_suite.adapter import Bridge
from rsi_coding_suite.ownership import OwnedProcesses


def test_absolute_bridge_timeout_includes_wedged_stdin_drain(tmp_path):
    class Stdin:
        def write(self,value):pass
        async def drain(self):await asyncio.Event().wait()
    class Stdout:
        reached=False
        async def readline(self):self.reached=True;return b'{}\n'
    async def run():
        proc=type('Fake',(),{'stdin':Stdin(),'stdout':Stdout()})()
        bridge=Bridge(proc,tmp_path,None)
        started=time.monotonic()
        with pytest.raises(TimeoutError):await asyncio.wait_for(bridge.exchange({},.01),.5)
        assert time.monotonic()-started<.2 and not proc.stdout.reached
    asyncio.run(run())


def test_cleanup_detached_python_child_after_direct_parent_exits(tmp_path):
    async def run():
        owned=OwnedProcesses(tmp_path/'owner')
        pidfile=tmp_path/'child'
        code="import subprocess,sys,pathlib;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c',code,str(pidfile),
            start_new_session=True,env=owned.environment())
        owned.register(proc.pid)
        await asyncio.wait_for(proc.wait(),5)
        child=int(pidfile.read_text());created=psutil.Process(child).create_time()
        bridge=Bridge(proc,tmp_path,None,owned)
        await bridge.close()
        assert proc.returncode is not None
        try:
            childproc=psutil.Process(child)
            assert childproc.create_time()!=created or childproc.status()==psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:pass
        assert json.loads((tmp_path/'owner/process-cleanup.json').read_text())['remaining_owned_pids']==[]
    asyncio.run(run())


def test_cancel_tool_reaps_nested_sessions_and_preserves_unrelated_process(tmp_path):
    async def run():
        owned=OwnedProcesses(tmp_path/'owner');pidfile=tmp_path/'child'
        unrelated=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(20)',start_new_session=True)
        code="import subprocess,sys,pathlib,time;sys.stdin.readline();p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(20)"
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c',code,str(pidfile),
            start_new_session=True,env=owned.environment(),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
        owned.register(proc.pid);bridge=Bridge(proc,tmp_path,None,owned)
        task=asyncio.create_task(bridge.exchange({},5))
        try:
            for _ in range(100):
                if pidfile.exists():break
                await asyncio.sleep(.02)
            assert pidfile.exists();child=int(pidfile.read_text());created=psutil.Process(child).create_time()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
            await bridge.close()
            assert proc.returncode is not None and unrelated.returncode is None
            try:
                childproc=psutil.Process(child)
                assert childproc.create_time()!=created or childproc.status()==psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:pass
        finally:
            await bridge.close()
            if unrelated.returncode is None:unrelated.kill()
            await unrelated.wait()
    asyncio.run(run())


def test_creation_identity_mismatch_never_signals_unrelated(tmp_path):
    owned=OwnedProcesses(tmp_path)
    owned.entries[psutil.Process().pid]=psutil.Process().create_time()-1
    assert owned.refresh()==[]
    import signal
    assert not owned.signal(psutil.Process().pid,signal.SIGSTOP)


def test_deadline_budget_is_shared_by_drain_and_read(tmp_path):
    class Stdin:
        def write(self,value):pass
        async def drain(self):await asyncio.sleep(.04)
    class Stdout:
        async def readline(self):await asyncio.sleep(.04);return b'{}\n'
    async def run():
        proc=type('Fake',(),{'stdin':Stdin(),'stdout':Stdout()})()
        with pytest.raises(TimeoutError):await Bridge(proc,tmp_path,None).exchange({},.06)
    asyncio.run(run())


def test_actual_full_pipe_request_times_out_and_reaps(tmp_path):
    async def run():
        owned=OwnedProcesses(tmp_path/'owner')
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(20)',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
            start_new_session=True,env=owned.environment())
        owned.register(proc.pid);bridge=Bridge(proc,tmp_path,None,owned)
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(bridge.exchange({'code':'x'*2_000_000},.03),1)
        finally:await bridge.close()
        assert proc.returncode is not None
    asyncio.run(run())


@pytest.mark.parametrize('entry_point',['owner','bridge'])
@pytest.mark.parametrize('cleanup_stage',[1,2,3])
def test_repeated_cancellation_finishes_cleanup_before_propagating(tmp_path,entry_point,cleanup_stage):
    async def run():
        owned=OwnedProcesses(tmp_path/'owner')
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c','import time;time.sleep(20)',
            start_new_session=True,env=owned.environment())
        owned.register(proc.pid)
        log=(tmp_path/'stderr').open('wb');bridge=Bridge(proc,tmp_path,log,owned)
        reached=asyncio.Event();original=owned.refresh;calls=0
        def observed_refresh():
            nonlocal calls
            calls+=1
            result=original()
            if calls==cleanup_stage:reached.set()
            return result
        owned.refresh=observed_refresh
        task=asyncio.create_task(owned.cleanup() if entry_point=='owner' else bridge.close())
        try:
            await asyncio.wait_for(reached.wait(),1)
            task.cancel()
            await asyncio.sleep(.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,2)
            # Cancellation does not escape until terminal signaling and receipt.
            assert json.loads((tmp_path/'owner/process-cleanup.json').read_text())['remaining_owned_pids']==[]
            await asyncio.wait_for(proc.wait(),1)
            if entry_point=='bridge':assert log.closed
        finally:
            await bridge.close()
    asyncio.run(run())


def test_detached_grandchild_registered_after_custom_environment_and_parent_exit(tmp_path):
    async def run():
        owned=OwnedProcesses(tmp_path/'owner');pidfile=tmp_path/'grandchild'
        inner="import subprocess,sys,pathlib;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],env={'PYTHONPATH':''},start_new_session=True);pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
        outer="import subprocess,sys;p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],env={'PYTHONPATH':''},start_new_session=True);p.wait()"
        proc=await asyncio.create_subprocess_exec(sys.executable,'-c',outer,inner,str(pidfile),
            start_new_session=True,env=owned.environment())
        owned.register(proc.pid);bridge=Bridge(proc,tmp_path,None,owned)
        try:
            await asyncio.wait_for(proc.wait(),5)
            grandchild=int(pidfile.read_text());created=psutil.Process(grandchild).create_time()
            await bridge.close()
            try:
                child=psutil.Process(grandchild)
                assert child.create_time()!=created or child.status()==psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:pass
            assert len(owned.entries)>=3
        finally:await bridge.close()
    asyncio.run(run())


def test_malformed_journal_rows_are_ignored(tmp_path):
    owned=OwnedProcesses(tmp_path)
    owned.ledger.write_text('[]\n'+json.dumps({'token':owned.token,'pid':1})+'\n{')
    assert owned.refresh()==[]
