"""Trainer callback invoked UNDER the native GroupStore's state flock.

No native/store calls, generation, scorer execution, model loading, network or
accelerator-lock acquisition occurs here. The caller prepares its learner and
GPU lease before entering native consume. Native rereads the returned latest
checkpoint before releasing its lock or authorizing own-group promotion.
"""
from copy import deepcopy
import math
from pathlib import Path
import uuid

from journal import SCHEMA,Checkpoints,atomic,digest,hex_id,locked,native_digest,read,require,sha

AUTH_SCHEMA='training-researcher-group8-consume-authorization-v1'


def prepare_optimizer_inputs(authorization,checkpoint):
    """Actual null exclusion, lag decision and GRPO normalization; no stub.

    This is called only with the trusted native facade's sealed authorization.
    It is NOT an agent/CLI JSON reward admission endpoint.
    """
    a=authorization
    require(a['schema']==AUTH_SCHEMA and hex_id(a['group_id']) and
            (hex_id(a['sealed_group_sha256'],64) or (a['sealed_group_sha256'] is None and not a['complete'])),
            'native group authorization required')
    require(a['bindings']==checkpoint['bindings'],'source/model/native/budget bindings differ')
    require(type(a['complete']) is bool and type(a['frozen_revision']) is int and
            type(a['current_revision']) is int and a['current_revision']>=a['frozen_revision']>=0,
            'nondecreasing native incumbent revisions required')
    require(hex_id(a['frozen_incumbent_sha256'],64) and hex_id(a['current_incumbent_sha256'],64),
            'native incumbent hashes required')
    lag=a['current_revision']-a['frozen_revision']
    if lag==0:require(a['current_incumbent_sha256']==a['frozen_incumbent_sha256'],'same revision changed source')
    require(len(a['sealed_attempt_hashes'])==len(a['rewards'])==len(a['exclusions'])==8,'eight slot records required')
    require(a['policy_state_sha256']==checkpoint['policy_state_sha256'],'sampled policy differs from loaded saved policy')
    require(hex_id(a['samples_sha256'],64) and hex_id(a['reset_checkpoint_sha256'],64),'sample/checkpoint hashes required')
    plan=dict(group_id=a['group_id'],sealed_group_sha256=a['sealed_group_sha256'],bindings=a['bindings'],
        batch_sha256=a['batch_sha256'],
        frozen_revision=a['frozen_revision'],frozen_incumbent_sha256=a['frozen_incumbent_sha256'],
        current_revision=a['current_revision'],current_incumbent_sha256=a['current_incumbent_sha256'],
        revision_lag=lag,stale_incumbent=lag==1,rewards=list(a['rewards']),exclusions=list(a['exclusions']),
        sealed_attempt_hashes=list(a['sealed_attempt_hashes']),samples_sha256=a['samples_sha256'],
        reset_checkpoint_sha256=a['reset_checkpoint_sha256'],policy_state_sha256=a['policy_state_sha256'])
    if 'native_outcome' in a:plan['native_outcome']=a['native_outcome']
    if 'candidate_source_hashes' in a:plan['candidate_source_hashes']=a['candidate_source_hashes']
    if a.get('native_action')=='discard':
        return dict(plan,action='terminal',reason=a['native_reason'],counted=0,advantages=[None]*8)
    if not a['complete']:
        return dict(plan,action='terminal',reason='discard_incomplete_group',counted=0,advantages=[None]*8)
    require(all(hex_id(x,64) for x in a['sealed_attempt_hashes']) and
            len(set(a['sealed_attempt_hashes']))==8,'distinct sealed native attempts required')
    counted=[]
    for slot,(reward,exclusion) in enumerate(zip(a['rewards'],a['exclusions'])):
        require(exclusion is None or isinstance(exclusion,str),'explicit null exclusion required')
        if reward is None:
            require(isinstance(exclusion,str) and exclusion,'null is infrastructure/excluded, never zero')
        else:
            require(type(reward) in (int,float) and math.isfinite(reward) and 0<=reward<=1
                    and exclusion is None,'eligible finite native score0..1 required')
            counted.append((slot,float(reward)))
    if lag>=2:
        return dict(plan,action='terminal',reason='discard_incumbent_lag_ge2',counted=len(counted),advantages=[None]*8)
    if len(counted)<2:
        return dict(plan,action='terminal',reason='discard_fewer_than_two_counted',counted=len(counted),advantages=[None]*8)
    mean=math.fsum(r for _,r in counted)/len(counted)
    denominator=len(counted) if a.get('native_action')=='update' else len(counted)-1
    std=math.sqrt(math.fsum((r-mean)**2 for _,r in counted)/denominator)
    if std==0:
        return dict(plan,action='terminal',reason='discard_zero_variance',counted=len(counted),advantages=[None]*8,
                    reward_mean=mean,reward_std=std)
    advantages=[None]*8
    for slot,reward in counted:advantages[slot]=(reward-mean)/(std if a.get('native_action')=='update' else std+1e-4)
    return dict(plan,action='optimizer_step',reason=None,counted=len(counted),advantages=advantages,
                reward_mean=mean,reward_std=std)


class TrainerCallback:
    """A loaded learner is injected; native GroupStore is the only caller.

    learner.step(plan, checkpoint, output_dir) must perform exactly one step,
    persist policy+Adam+RNG, and return the receipt contract in CONTRACT.md.
    Fault hooks exist only for explicit fixture bindings.
    """
    def __init__(self,checkpoints,learner,*,fault_hook=None):
        require(isinstance(checkpoints,Checkpoints),'trainer checkpoint store required')
        require(fault_hook is None or checkpoints.bindings['fixture_only'],'fault hooks are fixture-only')
        self.checkpoints=checkpoints;self.learner=learner;self.fault_hook=fault_hook

    def fault(self,point):
        if self.fault_hook:self.fault_hook(point)

    def result(self,checkpoint,checksum,gid,disposition):
        return dict(schema='training-researcher-group8-callback-result-v1',group_id=gid,disposition=disposition,
            checkpoint_path=checkpoint['checkpoint_path'],checkpoint_sha256=checksum,sequence=checkpoint['sequence'],
            policy_state_path=checkpoint['policy_state_path'],policy_state_sha256=checkpoint['policy_state_sha256'],
            consumed_group_ids=checkpoint['consumed_group_ids'],terminal_group_ids=checkpoint['terminal_group_ids'],
            bindings=checkpoint['bindings'])

    def __call__(self,authorization):
        # Native state lock is already held by the caller, before trainer.lock.
        # This lock order must never be reversed by a wrapper or recovery path.
        with locked(self.checkpoints.root/'trainer.lock'):
            checkpoint,checksum=self.checkpoints.latest()
            gid=authorization['group_id']
            if gid in checkpoint['consumed_group_ids']:
                require(checkpoint['consumed_rewards'][gid]['sealed_group_sha256']==authorization['sealed_group_sha256'],
                        'same consumed group id changed sealed reward evidence')
                return self.result(checkpoint,checksum,gid,'consumed')
            if gid in checkpoint['terminal_group_ids']:
                require(checkpoint['terminal_groups'][gid]['sealed_group_sha256']==authorization['sealed_group_sha256'],
                        'same terminal group id changed sealed evidence')
                return self.result(checkpoint,checksum,gid,'terminal')
            plan=prepare_optimizer_inputs(authorization,checkpoint)
            updated=deepcopy(checkpoint)
            if plan['action']=='terminal':
                terminal_dir=self.checkpoints.root/'transactions'/uuid.uuid4().hex
                terminal_dir.mkdir()
                atomic(terminal_dir/'discard-plan.json',plan)
                updated['terminal_group_ids'].append(gid)
                updated['terminal_groups'][gid]=dict(reason=plan['reason'],sealed_group_sha256=plan['sealed_group_sha256'],
                    plan_path=str(terminal_dir/'discard-plan.json'),plan_sha256=sha(terminal_dir/'discard-plan.json'),
                    optimizer_step_called=False,promotion_allowed=False)
                updated['group_outcomes'][gid]=plan.get('native_outcome',dict(action='discard',batch_sha256=plan['batch_sha256'],
                    sealed_group_sha256=plan['sealed_group_sha256'],reason=plan['reason']))
                updated.update(optimizer_step_completed=False,losses=dict(before=None,after=None),losses_before=None,losses_after=None)
                updated['tensor_hashes']['adapter_before']=checkpoint['tensor_hashes']['adapter_after']
                self.fault('before_terminal_checkpoint')
                saved,saved_hash=self.checkpoints.save(updated,checksum,fault_hook=self.fault_hook)
                self.fault('after_terminal_checkpoint')
                return self.result(saved,saved_hash,gid,'terminal')
            require(self.learner is not None and
                    self.learner.prepared_policy_state_sha256==checkpoint['policy_state_sha256'],
                    'learner must already be loaded from the latest committed policy+Adam state')
            transaction=self.checkpoints.root/'transactions'/uuid.uuid4().hex
            transaction.mkdir()
            plan_hash=digest(plan)
            atomic(transaction/'intent.json',dict(group_id=gid,plan=plan,plan_sha256=plan_hash,
                previous_checkpoint_sha256=checksum,consumption_authority=False))
            self.fault('before_optimizer')
            # No source/model loading, new lease, native scorer or GroupStore call
            # may be performed by this already-prepared learner's step method.
            receipt=self.learner.step(plan,checkpoint,transaction)
            self.fault('after_optimizer_before_receipt')
            self.verify_step(receipt,plan,plan_hash,checkpoint,transaction)
            atomic(transaction/'step-receipt.json',receipt)
            self.fault('after_receipt_before_checkpoint')
            updated.update(policy_state_path=receipt['policy_state_path'],policy_state_sha256=receipt['policy_state_sha256'],
                adapter_weights_sha256=receipt['adapter_after_sha256'],optimizer_calls=checkpoint['optimizer_calls']+1,
                step=checkpoint['step']+1,optimizer_step_completed=True,
                final_weights=dict(path=receipt['policy_state_path'],sha256=receipt['policy_state_sha256'],artifact_kind='lora-adam-rng'),
                tensor_hashes=dict(base=receipt['base_after_sha256'],adapter_before=receipt['adapter_before_sha256'],adapter_after=receipt['adapter_after_sha256']),
                optimizer_state_sha256=receipt['optimizer_state_sha256'],rng_state_sha256=receipt['rng_state_sha256'],
                losses=dict(before=receipt['loss_before'],after=receipt['loss_after']),losses_before=receipt['loss_before'],losses_after=receipt['loss_after'],
                eligible_changed_updates=checkpoint['eligible_changed_updates']+
                    int(receipt['adapter_before_sha256']!=receipt['adapter_after_sha256']))
            updated['consumed_group_ids'].append(gid)
            updated['group_outcomes'][gid]=plan.get('native_outcome',dict(action='update',batch_sha256=plan['batch_sha256'],
                sealed_group_sha256=plan['sealed_group_sha256']))
            updated['consumed_rewards'][gid]=dict(sealed_group_sha256=plan['sealed_group_sha256'],rewards=plan['rewards'],
                stale_incumbent=plan['stale_incumbent'],revision_lag=plan['revision_lag'],
                plan_path=str(transaction/'intent.json'),plan_sha256=sha(transaction/'intent.json'),
                step_receipt_path=str(transaction/'step-receipt.json'),step_receipt_sha256=sha(transaction/'step-receipt.json'),
                optimizer_loss=receipt['loss'],max_logprob_gap=receipt['max_logprob_gap'])
            self.fault('before_checkpoint_publish')
            saved,saved_hash=self.checkpoints.save(updated,checksum,fault_hook=self.fault_hook)
            self.fault('after_checkpoint_publish')
            return self.result(saved,saved_hash,gid,'consumed')

    def verify_step(self,receipt,plan,plan_hash,checkpoint,transaction):
        require(receipt['schema']=='training-researcher-group8-step-v1' and receipt['group_id']==plan['group_id']
                and receipt['plan_sha256']==plan_hash and receipt['bindings']==checkpoint['bindings']
                and receipt['optimizer_steps']==1,'optimizer receipt group/plan/source/count differs')
        kind='fixture' if checkpoint['bindings']['fixture_only'] else 'cuda_lora'
        require(receipt['runtime_kind']==kind,'fixture optimizer cannot authorize real model updates')
        require(receipt['previous_policy_state_sha256']==checkpoint['policy_state_sha256'], 'optimizer predecessor differs')
        output=Path(receipt['policy_state_path'])
        require(output==output.resolve() and output.is_relative_to(transaction.resolve()) and
                output.is_file() and not output.is_symlink() and sha(output)==receipt['policy_state_sha256'],
                'durable model/Adam/RNG blob missing, changed or outside transaction')
        require(receipt['base_before_sha256']==receipt['base_after_sha256']==checkpoint['base_weights_sha256']
                ==checkpoint['initial_base_weights_sha256'], 'frozen model weights changed')
        require(receipt['adapter_before_sha256']==checkpoint['adapter_weights_sha256'] and
                hex_id(receipt['adapter_after_sha256'],64),'adapter tensor lineage differs')
        require(type(receipt['loss']) in (int,float) and math.isfinite(receipt['loss']) and
                type(receipt['max_logprob_gap']) in (int,float) and math.isfinite(receipt['max_logprob_gap'])
                and 0<=receipt['max_logprob_gap']<=1e-4,'loss/own-sampler replay gap invalid')
        require(all(type(receipt[k]) in (int,float) and math.isfinite(receipt[k])
                    for k in ('loss_before','loss_after')),'actual before/after losses required')
        require(hex_id(receipt['optimizer_state_sha256'],64) and hex_id(receipt['rng_state_sha256'],64),
                'canonical Adam/RNG component hashes required')


class NativeCommitAdapter:
    """Exact GroupStore.consume(gid, commit, deadline_monotonic=...) callback.

    It maps the trusted native batch, preserves its outcome verbatim, and
    returns EXACTLY the six fields checked by the native consume finally block.
    """
    def __init__(self,callback):self.callback=callback

    def __call__(self,batch):
        checkpoint,_=self.callback.checkpoints.latest()
        require(batch['schema']=='training-group8-public-v1' and batch['batch_sha256']==
                native_digest({k:v for k,v in batch.items() if k!='batch_sha256'}),'native batch hash/schema differs')
        require(batch['bindings']['trainer']==checkpoint['bindings'] and
                batch['checkpoint_before_sha256']==sha(checkpoint['checkpoint_path']),'native checkpoint/bindings differ')
        samples=batch['samples']
        if batch['action']=='update':
            expected_policy=native_digest(dict(base=checkpoint['tensor_hashes']['base'],
                adapter=checkpoint['tensor_hashes']['adapter_after'],model=checkpoint['bindings']['policy_model_sha256']))
            require(samples['sampled_policy_sha256']==expected_policy,'sealed sampled model tensors differ from checkpoint')
        outcome={k:batch[k] for k in ('action','batch_sha256','sealed_group_sha256','counted_indices','reason')}
        outcome['consumed_rewards']=batch['counted_rewards']
        authorization=dict(schema=AUTH_SCHEMA,group_id=batch['group_id'],sealed_group_sha256=batch['sealed_group_sha256'],
            batch_sha256=batch['batch_sha256'],bindings=checkpoint['bindings'],frozen_revision=batch['incumbent_revision'],
            frozen_incumbent_sha256=batch['incumbent_sha256'],current_revision=batch['current_revision'],
            current_incumbent_sha256=batch['current_incumbent_sha256'],complete=batch['sealed_group_sha256'] is not None,
            sealed_attempt_hashes=batch['sealed_attempt_hashes'],rewards=batch['sealed_rewards'],
            exclusions=[('; '.join(reasons) or 'native excluded') if reward is None else None
                for reward,reasons in zip(batch['sealed_rewards'],batch['sealed_exclusions'])],
            policy_state_sha256=checkpoint['policy_state_sha256'],
            samples_sha256=samples['artifact']['sha256'] if samples else native_digest(dict(no_samples=batch['group_id'])),
            reset_checkpoint_sha256=batch['reset_checkpoint_sha256'],native_action=batch['action'],
            native_reason=batch['reason'],native_outcome=outcome)
        if samples:authorization['candidate_source_hashes']=samples['source_sha256']
        result=self.callback(authorization)
        return {k:result[k] for k in ('checkpoint_path','checkpoint_sha256','group_id','disposition','policy_state_path','policy_state_sha256')}


def promotion_disposition(checkpoints,gid):
    """Native submit guard reads checkpoint authority, never group rewards.

    True here means only "consumed and not discarded". Native candidate-specific
    reserved validation and champion CAS are STILL mandatory for promotion.
    """
    checkpoint,checksum=checkpoints.latest()
    return dict(group_id=gid,checkpoint_sha256=checksum,
        consumed=gid in checkpoint['consumed_group_ids'],terminal=gid in checkpoint['terminal_group_ids'],
        may_check_own_validation=(gid in checkpoint['consumed_group_ids'] and gid not in checkpoint['terminal_group_ids']),
        promotion_verified=False)
