"""Register Python descendants before user code and propagate native Popen hooks.

Trusted cooperative execution only. Direct syscalls, exec with cleared env, -I/-S,
non-Python daemon descendants, and deliberate registration evasion are not covered.
"""
import json
import os
from pathlib import Path
import subprocess

LEDGER=os.environ.get('RSI_OWNED_LEDGER')
TOKEN=os.environ.get('RSI_OWNED_TOKEN')
SITE=str(Path(__file__).resolve().parent)
if LEDGER and TOKEN:
    import psutil
    def register(pid):
        try:
            process=psutil.Process(pid)
            entry={'pid':pid,'created':process.create_time(),'token':TOKEN}
            handle=os.open(LEDGER,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
            try:os.write(handle,(json.dumps(entry)+'\n').encode())
            finally:os.close(handle)
        except (OSError,psutil.Error):pass
    register(os.getpid())
    original=subprocess.Popen
    class RegisteredPopen(original):
        def __init__(self,*args,**kwargs):
            # Kernel's worker resets PYTHONPATH: preserve its path while keeping
            # registration active for new Python sessions and their descendants.
            supplied=kwargs.get('env')
            env=dict(os.environ if supplied is None else supplied)
            env['RSI_OWNED_LEDGER']=LEDGER;env['RSI_OWNED_TOKEN']=TOKEN
            env['PYTHONPATH']=SITE+os.pathsep+env.get('PYTHONPATH','')
            kwargs['env']=env
            super().__init__(*args,**kwargs)
            register(self.pid)
    subprocess.Popen=RegisteredPopen
    if hasattr(os,'register_at_fork'):
        os.register_at_fork(after_in_child=lambda:register(os.getpid()))
