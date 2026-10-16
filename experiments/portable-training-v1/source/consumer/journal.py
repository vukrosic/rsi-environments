"""Atomic trainer checkpoints. Stdlib, no model imports or native group writes.

One shared native league owns opponents. These files store trainer transactions,
not another incumbent database. Only complete checkpoint JSON files authorize
consumption; journal consumed flags and dangling optimizer outputs never do.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

SCHEMA='training-researcher-group8-trainer-v1'


def require(value,reason):
    if not value:raise ValueError(reason)


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def native_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(filename):
    value=hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):value.update(chunk)
    return value.hexdigest()


def read(filename):
    filename=Path(filename)
    require(filename.is_file() and not filename.is_symlink() and filename.stat().st_size<=12_000_000,
            'bounded regular metadata required')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate metadata key')
            result[key]=value
        return result
    def reject(value):raise ValueError('nonfinite metadata')
    result=json.loads(filename.read_bytes(),object_pairs_hook=pairs,parse_constant=reject)
    require(isinstance(result,dict),'metadata object required')
    return result


def sync(folder):
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic(filename,value,*,replace=False):
    filename=Path(filename)
    require(not filename.is_symlink(),'metadata symlink rejected')
    require(replace or not filename.exists(),'immutable metadata already exists')
    temporary=filename.with_name('.'+filename.name+'-'+uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    # All writers cooperate with the lock. Rename publishes only the complete,
    # fsynced document; a dotfile left before rename is not a saved checkpoint.
    os.replace(temporary,filename)
    sync(filename.parent)


def hex_id(value,length=32):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


@contextmanager
def locked(filename):
    require(not Path(filename).is_symlink(),'lock symlink rejected')
    with Path(filename).open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def bindings_valid(value):
    needed={'trainer_id','source_sha256','model_sha256','budget_sha256','native_source_sha256','policy_config_sha256','fixture_only'}
    require(needed<=set(value) and hex_id(value['trainer_id']) and type(value['fixture_only']) is bool,
            'complete trainer/source/model/budget bindings required')
    require(all(hex_id(value[k],64) for k in needed-{'trainer_id','fixture_only'}),'SHA256 binding required')
    for native,internal in (('run_id','trainer_id'),('policy_model_sha256','model_sha256'),('trainer_budget_sha256','budget_sha256')):
        if native in value:require(value[native]==value[internal],'native/internal binding alias differs: '+native)
    if not value['fixture_only']:
        require({'run_id','checkpoint_pointer','trainer_source_sha256','policy_model_sha256','trainer_budget_sha256','base_weights'}<=set(value),
                'production native trainer binding is incomplete')


class Checkpoints:
    """Stable latest.json pointer is the atomic checkpoint commit point.

    Artifacts and an immutable manifest are durable first. Replacing latest.json
    commits them. Higher orphan manifests are uncommitted, never authority.
    A broken published pointer fails closed; never scan backward or forward.
    """
    def __init__(self,root,bindings):
        bindings_valid(bindings)
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.files=self.root/'checkpoints';self.files.mkdir(exist_ok=True)
        (self.root/'transactions').mkdir(exist_ok=True)
        self.pointer=self.root/'latest.json'
        self.bindings=bindings

    def latest(self):
        # The published pointer is the sole commit point. Higher orphan
        # manifests are uncommitted and never authorize consumption.
        pointer=read(self.pointer)
        require(set(pointer)=={'checkpoint_path','checkpoint_sha256'},'exact published pointer required')
        filename=Path(pointer['checkpoint_path']);expected=pointer['checkpoint_sha256']
        newer=None;latest=None;depth=0
        while filename is not None:
            require(filename==filename.resolve() and filename.parent==self.files and sha(filename)==expected,
                    'published checkpoint manifest missing, mutated or outside trainer')
            value=read(filename)
            require(value['schema']=='training-group8-checkpoint-v1' and value['checkpoint_path']==str(filename)
                    and value['bindings']==self.bindings and value['run_id']==self.bindings['trainer_id']
                    and value['checkpoint_index']==value['sequence'],'checkpoint index/source/model/run differs')
            require(value['step']==value['optimizer_calls'] and type(value['optimizer_step_completed']) is bool,
                    'checkpoint optimizer-step semantics differ')
            ids=value['consumed_group_ids'];terminal=value['terminal_groups']
            require(len(ids)==len(set(ids)) and not set(ids)&set(terminal)
                    and len(value['terminal_group_ids'])==len(set(value['terminal_group_ids']))
                    and set(value['terminal_group_ids'])==set(terminal), 'terminal/consumed IDs differ')
            require(set(value['consumed_rewards'])==set(ids) and
                    set(value['group_outcomes'])==set(ids)|set(terminal),'checkpoint group evidence differs')
            if newer:
                require(newer['checkpoint_index']==value['checkpoint_index']+1
                        and set(ids)<=set(newer['consumed_group_ids']) and set(terminal)<=set(newer['terminal_groups']),
                        'checkpoint erased history or index chain differs')
            else:latest=value
            depth+=1;require(depth<=10000,'bounded checkpoint chain exceeded')
            newer=value
            previous=value['previous_checkpoint_path'];expected=value['previous_checkpoint_sha256']
            filename=Path(previous) if previous is not None else None
        require(newer['checkpoint_index']==0 and newer['previous_checkpoint_sha256'] is None,'initial checkpoint missing')
        require(sha(latest['policy_state_path'])==latest['policy_state_sha256'] and
                latest['final_weights']['path']==latest['policy_state_path'] and
                latest['final_weights']['sha256']==latest['policy_state_sha256'] and
                latest['final_weights']['artifact_kind']=='lora-adam-rng', 'published policy/Adam/RNG state changed')
        require(sha(latest['base_weights']['path'])==latest['base_weights']['sha256'],'frozen base artifact changed')
        for value in latest['consumed_rewards'].values():
            require(sha(value['plan_path'])==value['plan_sha256'] and
                    sha(value['step_receipt_path'])==value['step_receipt_sha256'],'consumed reward sidecar changed')
        for value in latest['terminal_groups'].values():
            require(sha(value['plan_path'])==value['plan_sha256'],'terminal discard proof changed')
        return latest,pointer['checkpoint_sha256']

    def publish(self,state,*,fault_hook=None):
        value=dict(state)
        filename=self.files/('%08d-%s.json'%(value['checkpoint_index'],uuid.uuid4().hex))
        value['checkpoint_path']=str(filename)
        require(len(encoded(value))<=950000,'checkpoint would exceed native1MB metadata cap; stop before publication')
        atomic(filename,value)
        if fault_hook:fault_hook('manifest_saved_before_pointer')
        checksum=sha(filename)
        atomic(self.pointer,dict(checkpoint_path=str(filename),checkpoint_sha256=checksum),replace=self.pointer.exists())
        return value,checksum

    def initialize(self,policy_state,base_sha256,adapter_sha256,base_artifact,*,optimizer_state_sha256=None,rng_state_sha256=None):
        with locked(self.root/'trainer.lock'):
            require(not self.pointer.exists(),'trainer already initialized')
            require(hex_id(base_sha256,64) and hex_id(adapter_sha256,64),'initial tensor hashes required')
            require(sha(base_artifact['path'])==base_artifact['sha256'],'base file binding differs')
            require(self.bindings['fixture_only'] or (hex_id(optimizer_state_sha256,64) and hex_id(rng_state_sha256,64)),
                    'initial canonical Adam/RNG hashes required')
            state=dict(schema='training-group8-checkpoint-v1',run_id=self.bindings['trainer_id'],checkpoint_index=0,sequence=0,step=0,
                previous_checkpoint_path=None,previous_checkpoint_sha256=None,bindings=self.bindings,
                policy_state_path=str(Path(policy_state).resolve()),policy_state_sha256=sha(policy_state),
                base_weights=dict(base_artifact),final_weights=dict(path=str(Path(policy_state).resolve()),sha256=sha(policy_state),artifact_kind='lora-adam-rng'),
                state_components=['policy','adam','rng'],tensor_hashes=dict(base=base_sha256,adapter_before=adapter_sha256,adapter_after=adapter_sha256),
                optimizer_state_sha256=optimizer_state_sha256 or digest(dict(fixture='adam')),
                rng_state_sha256=rng_state_sha256 or digest(dict(fixture='rng')),
                parent_checkpoint_sha256=None,parent_state_sha256=None,
                base_weights_sha256=base_sha256,adapter_weights_sha256=adapter_sha256,
                initial_base_weights_sha256=base_sha256,initial_adapter_weights_sha256=adapter_sha256,
                optimizer_calls=0,optimizer_step_completed=False,eligible_changed_updates=0,
                losses=dict(before=None,after=None),losses_before=None,losses_after=None,consumed_group_ids=[],consumed_rewards={},
                terminal_group_ids=[],terminal_groups={},group_outcomes={})
            return self.publish(state)

    def save(self,state,expected_hash,*,fault_hook=None):
        current,current_hash=self.latest()
        require(current_hash==expected_hash,'checkpoint advanced; uncommitted optimizer output cannot publish')
        value=dict(state,checkpoint_index=current['checkpoint_index']+1,sequence=current['sequence']+1,previous_checkpoint_path=current['checkpoint_path'],
                   previous_checkpoint_sha256=current_hash,parent_checkpoint_sha256=current_hash,parent_state_sha256=current['policy_state_sha256'])
        require(value['bindings']==self.bindings and sha(value['policy_state_path'])==value['policy_state_sha256'],
                'new model/optimizer state or bindings missing/changed')
        return self.publish(value,fault_hook=fault_hook)


