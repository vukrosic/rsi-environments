"""Concrete entry points over the pinned native GroupStore callback API.

No native group journal is written here. No GPU lease, model load, generation,
native scoring or host connection is performed by these transaction functions.
"""
import importlib
import sys
from pathlib import Path

from journal import read,require,sha
from transaction import NativeCommitAdapter,TrainerCallback

HERE=Path(__file__).resolve().parent
NATIVE_SHA='f469c648e6b240fe0a390556e937637f8beba6f93e230b8bdf287b01ee2049e7'


def load_store(native_root,state):
    root=Path(native_root).resolve()
    require(sha(HERE/'NATIVE-MANIFEST.json')==NATIVE_SHA,'exact native group8 source manifest required')
    manifest=read(HERE/'NATIVE-MANIFEST.json')
    require(len(manifest['files'])==20,'exact20 runtime-only native sources required')
    for item in manifest['files']:
        file=root/item['path']
        require(file.is_file() and not file.is_symlink() and sha(file)==item['sha256'],'native frozen source changed: '+item['path'])
    expected={'group_state':root/'training-env/group8-v1/group_state.py',
        'group_adapter':root/'training-env/group8-v1/group_adapter.py',
        'training_league':root/'training-env/training_league.py','league_adapter':root/'training-env/league_adapter.py',
        'suite':root/'suite/__init__.py','suite.league.controller':root/'suite/league/controller.py'}
    for name,file in expected.items():
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve()==file,'mixed native module origin: '+name)
    sys.path[:0]=[str(root/'training-env/group8-v1'),str(root/'training-env'),str(root)]
    module=importlib.import_module('group_state')
    for name,file in expected.items():
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve()==file,'native import origin differs: '+name)
    return module.GroupStore(state)


def consume_group(native_root,state,checkpoints,loaded_learner,gid,*,deadline_monotonic):
    """Model/Adam/RNG and the admitted GPU lease must already be prepared.

    GroupStore holds the existing state flock across its lag check, this
    callback, checkpoint publication and native verification. No extra flock
    surrounds this call. No native method runs inside TrainerCallback.
    """
    store=load_store(native_root,state)
    callback=NativeCommitAdapter(TrainerCallback(checkpoints,loaded_learner))
    return store.consume(gid,callback,deadline_monotonic=deadline_monotonic)


def recover_quiescent(native_root,state,checkpoints,*,deadline_monotonic,old_job_confirmed_stopped=False):
    """Only the executor can establish this precondition; never infer it from a timeout.

    Terminal groups get metadata-only checkpoint commits. Complete sealed groups
    are returned for later consumption after reloading the committed model/RNG.
    No new scoring or optimizer step occurs in recovery.
    """
    require(old_job_confirmed_stopped is True,'quiescent recovery needs executor confirmation old owned job stopped')
    store=load_store(native_root,state)
    pending=store.recover_groups()
    callback=NativeCommitAdapter(TrainerCallback(checkpoints,None))
    committed=[]
    for gid in pending['terminal_checkpoint_pending']:
        committed.append(store.consume(gid,callback,deadline_monotonic=deadline_monotonic))
    return dict(reusable_complete=pending['reusable_complete'],terminal_commits=committed,
                already_handled=pending['already_handled'],rescored=False)
