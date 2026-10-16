"""Native eight-candidate state facade; the external trainer owns all updates.

Only this facade writes groups/. Latest saved trainer checkpoint IDs, never a
group consumed flag, are consumption authority. Shared League remains artifact,
attempt, execution, native proof, history and champion CAS authority.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from training_league import TrainingLeague
from suite.league.controller import League, read, write, digest, sha
import league_adapter as native
from group_adapter import CONTRACT, OBJECTIVE, primitive, read_aa, promotion_gate


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def valid_id(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid group/round/attempt id')
    return value


def artifact(value):
    if (not isinstance(value, dict) or not {'path', 'sha256'} <= set(value)
            or set(value) - {'path', 'sha256', 'artifact_kind'}
            or not Path(value['path']).is_absolute() or file_sha(value['path']) != value['sha256']):
        raise ValueError('checkpoint weight/source artifact changed')


def sealed_group_payload(group):
    return {k: group[k] for k in ('schema', 'group_id', 'lane_id', 'incumbent', 'round_ids',
        'binding', 'binding_sha256', 'samples', 'attempts', 'sealed_rewards', 'sealed_exclusions')}


class GroupStore(TrainingLeague):
    @staticmethod
    def adapter_for(native_rounds):
        """Pin this public-source successor's exact runtime-only closure."""
        manifest = HERE / 'RUNTIME-MANIFEST.json'
        closure = read(manifest)
        expected = set(['training-env/' + name for name in native.NATIVE])
        expected.update(['training-env/league_adapter.py', 'training-env/training_league.py',
            'training-env/group8-v1/group_state.py', 'training-env/group8-v1/group_adapter.py',
            'suite/__init__.py', 'suite/records.py', 'suite/resources.py',
            'suite/league/__init__.py', 'suite/league/controller.py', 'suite/league/episode.py'])
        if (closure.get('schema') != 'training-group8-public-runtime-v1'
                or len(closure.get('files', [])) != 20
                or {item['path'] for item in closure['files']} != expected):
            raise ValueError('exact public runtime-only closure required')
        paths = []
        for item in closure['files']:
            path = HERE.parent.parent / item['path']
            if sha(path) != item['sha256']:
                raise ValueError('public runtime source changed')
            paths.append(path)
        paths.append(manifest)
        for value in native_rounds:
            paths += [Path(value['aa_path']).resolve(), Path(value['data_dir']).resolve() / 'manifest.json']
        return dict(argv=[sys.executable, str(HERE / 'group_adapter.py'), '--round', '{round}',
            '--candidate', '{candidate}', '--output', '{output}'], cwd=str(HERE.parent),
            trusted_files=sorted({str(p.resolve()) for p in paths}))

    def group(self, gid):
        value = read(self.root / 'groups' / (valid_id(gid) + '.json'))
        if (value.get('schema') != CONTRACT or value.get('group_id') != gid
                or value.get('record_sha256') != digest({k: v for k, v in value.items() if k != 'record_sha256'})
                or value.get('binding_sha256') != digest(value['binding'])):
            raise ValueError('group record/binding changed')
        if value.get('sealed_group_sha256') and value['sealed_group_sha256'] != digest(sealed_group_payload(value)):
            raise ValueError('sealed group changed')
        return value

    def save_group(self, value):
        value = {k: v for k, v in value.items() if k != 'record_sha256'}
        value['record_sha256'] = digest(value)
        write(self.root / 'groups' / (valid_id(value['group_id']) + '.json'), value)

    def moving_metadata(self, rid):
        folder = self.root / 'rounds' / valid_id(rid)
        if sha(folder / 'round.json') != read(folder / 'seal.json')['round_sha256']:
            raise ValueError('round mutated')
        value = read(folder / 'round.json')
        if sha(folder / 'native.json') != value['native_file_sha256']:
            raise ValueError('native round mutated')
        if value['native_round'].get('group_contract') != CONTRACT:
            return None
        return self.round(rid)

    @staticmethod
    def validate_native(value, files, adapter):
        TrainingLeague.validate_native(primitive(value), files, adapter)
        if (value.get('group_contract') != CONTRACT or value.get('objective_name') != OBJECTIVE
                or type(value.get('group_slot')) is not int or not 0 <= value['group_slot'] < 8
                or len(value['seeds']) < 3 or len(value['validation_seeds']) < 3
                or type(value.get('noise_gate')) not in (int, float)
                or not math.isfinite(value['noise_gate']) or value['noise_gate'] < .02):
            raise ValueError('explicit group8 successor and at least three seeds per partition required')
        valid_id(value['group_id'])
        required = [HERE / 'group_state.py', HERE / 'group_adapter.py']
        if any(files.get(str(p.resolve())) != sha(p) for p in required):
            raise ValueError('group source closure not pinned')
        if value.get('source_bindings') != files:
            # The AA JSON and data manifest are included too; no candidate may
            # supply the executed source map or alter it during a group.
            raise ValueError('group source/asset bindings differ from trusted registration')
        if files.get(str(Path(value['aa_path']).resolve())) != value['aa_sha256']:
            raise ValueError('A/A result is not pinned')
        aa = read_aa(value['aa_path'], value)
        if aa['noise_gate'] != value.get('noise_gate'):
            raise ValueError('empirical noise gate differs')

    @staticmethod
    def verify_pair_evidence(r, receipt, output):
        value = r['native_round']
        item = receipt['native_comparison']
        path = (Path(output) / item['path']).resolve()
        if not path.is_relative_to(Path(output).resolve()) or sha(path) != item['sha256']:
            raise ValueError('derived native comparison changed or escaped')
        inner = native.read_json(path)
        inner_round = native.read_json(Path(output) / 'paired-native-round.json')
        if inner_round != primitive(value) or inner.get('round_sha256') != sha(Path(output) / 'paired-native-round.json'):
            raise ValueError('derived native round changed')
        TrainingLeague.verify_pair_evidence(dict(r, native_round=inner_round), inner, path.parent)
        expected = dict(inner)
        gate = promotion_gate(inner, path.parent, value['noise_gate'])
        expected.update(objective_name=OBJECTIVE, round_sha256=r['native_file_sha256'],
            group_contract=CONTRACT, group_id=value['group_id'], group_slot=value['group_slot'],
            group_binding_sha256=value['group_binding_sha256'], aa_sha256=value['aa_sha256'],
            noise_gate=value['noise_gate'], source_bindings=value['source_bindings'],
            native_comparison=dict(path='native/comparison.json', sha256=sha(path)),
            native_receipts=[dict(x, path='native/' + x['path']) for x in inner['native_receipts']],
            promotion_ready=gate, validation=dict(inner['validation'], passed=gate))
        if receipt != expected:
            raise ValueError('group envelope differs from independent native comparison')

    def evaluate(self, r, candidate_sha, receipt, output):
        self.reject_reused_seeds(r['native_round'], exclude_round=r['round_id'])
        result = League.evaluate(r, candidate_sha, receipt, output)
        self.verify_pair_evidence(r, receipt, output)
        if candidate_sha == r['champion']['source_sha256']:
            result.update(score=None, promotion_eligible=False)
            result['exclusion_reasons'].append('candidate no-op excluded')
        # Measurement intentionally does not compare the CURRENT revision.
        # Lag is applied once at consumption; sealed rewards are never rescored.
        return result

    def latest_checkpoint(self, binding):
        required = {'run_id', 'checkpoint_pointer', 'trainer_source_sha256', 'policy_model_sha256',
                    'trainer_budget_sha256', 'base_weights'}
        if not required <= set(binding):
            raise ValueError('trainer/model/budget/source/base bindings required')
        if (not isinstance(binding['run_id'], str) or not binding['run_id']
                or not Path(binding['checkpoint_pointer']).is_absolute()
                or not isinstance(binding['trainer_source_sha256'], dict) or not binding['trainer_source_sha256']
                or any(not isinstance(binding[k], str) or len(binding[k]) != 64
                    or any(c not in '0123456789abcdef' for c in binding[k])
                    for k in ('policy_model_sha256', 'trainer_budget_sha256'))):
            raise ValueError('absolute pointer and nonempty source/model/budget bindings required')
        for path, checksum in binding['trainer_source_sha256'].items():
            if not Path(path).is_absolute() or file_sha(path) != checksum:
                raise ValueError('trainer source changed')
        artifact(binding['base_weights'])
        # latest.json is the atomic COMMIT point. Higher orphan manifests and
        # temp files are uncommitted and never authorize reward consumption.
        pointer = read(binding['checkpoint_pointer'])
        path = Path(pointer['checkpoint_path'])
        if not path.is_absolute() or file_sha(path) != pointer['checkpoint_sha256']:
            raise ValueError('published checkpoint pointer is broken; no fallback')
        checksum = pointer['checkpoint_sha256']
        ckpt = native.read_json(path)
        if (ckpt.get('schema') != 'training-group8-checkpoint-v1'
                or ckpt.get('run_id') != binding['run_id'] or ckpt.get('bindings') != binding
                or type(ckpt.get('step')) is not int or ckpt['step'] < 0
                or type(ckpt.get('checkpoint_index')) is not int or ckpt['checkpoint_index'] < 0):
            raise ValueError('checkpoint schema/run/source/model/budget binding differs')
        consumed, terminal = ckpt['consumed_group_ids'], ckpt['terminal_group_ids']
        if (not isinstance(consumed, list) or not isinstance(terminal, list)
                or len(set(consumed)) != len(consumed) or len(set(terminal)) != len(terminal)
                or set(consumed) & set(terminal) or set(ckpt['group_outcomes']) != set(consumed + terminal)):
            raise ValueError('checkpoint consumed/terminal group sets differ')
        for gid in consumed + terminal:
            valid_id(gid)
        artifact(ckpt['base_weights'])
        artifact(ckpt['final_weights'])
        if (ckpt['base_weights'] != binding['base_weights']
                or ckpt['base_weights'].get('artifact_kind') != 'pretrained-base'
                or ckpt['final_weights'].get('artifact_kind') != 'lora-adam-rng'
                or ckpt.get('state_components') != ['policy', 'adam', 'rng']):
            raise ValueError('pinned base and explicit adapter/Adam/RNG state blob required')
        hashes = [ckpt.get('tensor_hashes', {}).get(k) for k in ('base', 'adapter_before', 'adapter_after')]
        hashes += [ckpt.get('optimizer_state_sha256'), ckpt.get('rng_state_sha256')]
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('separate canonical base/adapter/Adam/RNG hashes required')
        return ckpt, checksum

    @staticmethod
    def verify_committed_group(group, checkpoint):
        gid = group['group_id']
        batch = group.get('prepared_batch')
        if not batch:
            raise ValueError('checkpoint group outcome lacks native prepared batch')
        if batch['batch_sha256'] != digest({k: v for k, v in batch.items() if k != 'batch_sha256'}):
            raise ValueError('native batch authorization changed')
        expected = dict(action=batch['action'], batch_sha256=batch['batch_sha256'],
            sealed_group_sha256=batch['sealed_group_sha256'], counted_indices=batch['counted_indices'],
            consumed_rewards=batch['counted_rewards'], reason=batch['reason'])
        target = 'consumed_group_ids' if batch['action'] == 'update' else 'terminal_group_ids'
        if (gid not in checkpoint[target] or checkpoint['group_outcomes'].get(gid) != expected
                or batch['sealed_group_sha256'] != group['sealed_group_sha256']):
            raise ValueError('checkpoint group outcome does not bind sealed rewards')

    def reset(self, native_rounds, adapter, trainer_binding):
        if not isinstance(native_rounds, list) or len(native_rounds) != 8:
            raise ValueError('exactly eight native rounds required')
        timeouts = [value['comparison_timeout_seconds'] for value in native_rounds]
        if (any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in timeouts)
                or math.fsum(timeouts) > 18000):
            raise ValueError('group native deadlines must total at most five hours within the six-hour job')
        checkpoint, checkpoint_sha = self.latest_checkpoint(trainer_binding)
        gid = uuid.uuid4().hex
        seeds = [s for value in native_rounds for field in ('seeds', 'validation_seeds') for s in value[field]]
        if len(seeds) != len(set(seeds)):
            raise ValueError('all candidate reward and own validation seeds must be disjoint')
        if len({digest(aa_identity_fields(v)) for v in native_rounds}) != 1:
            raise ValueError('one group must freeze the same incumbent/runtime/model/data/budget')
        files = {str(Path(p).resolve()): sha(p) for p in adapter['trusted_files']}
        binding = dict(trainer=trainer_binding, task_sources=files,
            task_model_sha256=native_rounds[0]['model_identity'], task_budget_sha256=digest(native_rounds[0]['budget']),
            rollout_checkpoint_sha256=checkpoint_sha, rollout_weights=checkpoint['final_weights'],
            rollout_policy_sha256=digest(dict(base=checkpoint['tensor_hashes']['base'],
                adapter=checkpoint['tensor_hashes']['adapter_after'], model=trainer_binding['policy_model_sha256'])))
        rounds = []
        # Freeze without running adapters. If a concurrent CAS lands during
        # reset, abort rather than returning a mixed-opponent group. Orphan
        # rounds have no registered group and cannot execute or promote.
        for slot, value in enumerate(native_rounds):
            value = dict(value, objective_name=OBJECTIVE, group_contract=CONTRACT, group_id=gid,
                group_slot=slot, group_binding_sha256=digest(binding), source_bindings=files)
            self.reject_reused_seeds(value)
            frozen = League.freeze(self, value, adapter, max_actions=1,
                timeout_seconds=value['comparison_timeout_seconds'])
            if rounds and frozen['champion'] != rounds[0]['champion']:
                raise ValueError('incumbent changed during group reset; no group registered')
            rounds.append(frozen)
        with self.locked():
            champion = read(self.root / 'champions' / (rounds[0]['lane_id'] + '.json'))
            if champion != rounds[0]['champion']:
                raise ValueError('incumbent changed before atomic group reset')
            value = dict(schema=CONTRACT, group_id=gid, lane_id=rounds[0]['lane_id'], incumbent=champion,
                round_ids=[r['round_id'] for r in rounds], binding=binding, binding_sha256=digest(binding),
                samples=None, attempts=[None] * 8, sealed_rewards=[None] * 8, sealed_exclusions=[None] * 8,
                consumed=False, phase='measuring',
                sealed_group_sha256=None, disposition=None)
            self.save_group(value)
        return self.observation(gid)

    def bind_samples(self, gid, sample_file, sampled_policy_sha256):
        """Seal eight trainer-generated samples; no generation occurs here."""
        artifact(sample_file)
        path = Path(sample_file['path'])
        if path.stat().st_size > 9 * 1024 * 1024:
            raise ValueError('sample file exceeds bounded group source size')
        samples = json.loads(path.read_text())
        with self.locked():
            group = self.group(gid)
            if group['phase'] != 'measuring' or group['samples'] is not None or any(group['attempts']):
                raise ValueError('samples already sealed or group closed')
            if (samples.get('schema') != 'training-group8-samples-v1' or samples.get('group_id') != gid
                    or samples.get('reset_checkpoint_sha256') != group['binding']['rollout_checkpoint_sha256']
                    or sampled_policy_sha256 != group['binding']['rollout_policy_sha256']
                    or samples.get('sampled_policy_sha256') != sampled_policy_sha256
                    or len(samples.get('samples', [])) != 8):
                raise ValueError('samples differ from group/reset checkpoint/policy')
            source_hashes = []
            for slot, sample in enumerate(samples['samples']):
                source = sample['source_text']
                if type(sample.get('slot')) is not int or sample['slot'] != slot or not isinstance(source, str):
                    raise ValueError('eight ordered source-text samples required')
                if len(source.encode()) > 1_048_576:
                    raise ValueError('candidate source exceeds native limit')
                source_hashes.append(hashlib.sha256(source.encode()).hexdigest())
            group['samples'] = dict(artifact=sample_file, sampled_policy_sha256=sampled_policy_sha256,
                                    source_sha256=source_hashes)
            self.save_group(group)
        return dict(group_id=gid, sample_file_sha256=sample_file['sha256'])

    def observation(self, gid):
        group = self.group(gid)
        return dict(group_id=gid, group_contract=CONTRACT, incumbent_revision=group['incumbent']['revision'],
            incumbent_sha256=group['incumbent']['source_sha256'], group_size=8,
            reset_checkpoint_sha256=group['binding']['rollout_checkpoint_sha256'],
            sampled_policy_sha256=group['binding']['rollout_policy_sha256'],
            completed_slots=[i for i, item in enumerate(group['attempts']) if item is not None],
            terminal=group['phase'] != 'measuring', binding_sha256=group['binding_sha256'])

    def execute(self, rid, candidate):
        with self.locked():
            r = self.round(rid)
            group = self.group(r['native_round']['group_id'])
            slot = r['native_round']['group_slot']
            if (group['phase'] != 'measuring' or group['round_ids'][slot] != rid
                    or group['binding_sha256'] != r['native_round']['group_binding_sha256']):
                raise ValueError('group is closed or round is not registered')
        return League.execute(self, rid, candidate)

    def checked_attempt(self, group, slot, aid):
        rid = group['round_ids'][slot]
        r = self.round(rid)
        if (r['champion'] != group['incumbent'] or r['native_round']['group_id'] != group['group_id']
                or r['native_round']['group_slot'] != slot
                or r['native_round']['group_binding_sha256'] != group['binding_sha256']):
            raise ValueError('attempt does not belong to this frozen group/slot')
        self.check_sources(r)
        folder = self.root / 'rounds' / rid / 'attempts' / valid_id(aid)
        checksum = sha(folder / 'attempt.json')
        if checksum != read(folder / 'seal.json')['attempt_sha256']:
            raise ValueError('sealed attempt changed')
        attempt = read(folder / 'attempt.json')
        if sha(attempt['candidate_path']) != attempt['candidate_sha256']:
            raise ValueError('candidate source changed')
        for relative, expected in attempt.get('proof', {}).items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder.resolve()) or file_sha(path) != expected:
                raise ValueError('attempt proof changed or escaped')
        if 'receipt' in attempt:
            checked = self.evaluate(r, attempt['candidate_sha256'], attempt['receipt'], folder / 'output')
            if any(checked[k] != attempt.get(k) for k in ('score', 'status', 'promotion_eligible')):
                raise ValueError('sealed attempt verdict differs')
        elif attempt.get('status') not in ('error', 'blocked') or attempt.get('score') is not None:
            raise ValueError('counted score requires a native receipt')
        return attempt, checksum

    def measure(self, gid, slot, source_text):
        if type(slot) is not int or not 0 <= slot < 8 or not isinstance(source_text, str):
            raise ValueError('slot and source text required')
        group = self.group(gid)
        if (not group['samples'] or hashlib.sha256(source_text.encode()).hexdigest()
                != group['samples']['source_sha256'][slot]):
            raise ValueError('source text differs from sealed sampled completion')
        artifact(group['samples']['artifact'])
        if not source_text:
            source_text = "# No code supplied by the policy.\nraise SyntaxError('missing candidate code')\n"
        # Snapshot source text ourselves, never trust an agent-selected path.
        with self.locked():
            source, _ = self.artifact_text(source_text)
        result = self.execute(group['round_ids'][slot], source)
        with self.locked():
            group = self.group(gid)
            if group['phase'] != 'measuring' or group['attempts'][slot] is not None:
                raise ValueError('group dropped or slot already sealed')
            attempt, checksum = self.checked_attempt(group, slot, result['attempt_id'])
            group['attempts'][slot] = dict(attempt_id=result['attempt_id'], attempt_sha256=checksum)
            group['sealed_rewards'][slot] = attempt['score']
            group['sealed_exclusions'][slot] = attempt.get('exclusion_reasons', [])
            if all(item is not None for item in group['attempts']):
                group['phase'] = 'complete'
                group['sealed_group_sha256'] = digest(sealed_group_payload(group))
            self.save_group(group)
        return dict(group_id=gid, slot=slot, attempt_id=result['attempt_id'],
                    status=result['status'], reward=result['score'])

    def artifact_text(self, text):
        data = text.encode()
        if not 0 < len(data) <= 1_048_576:
            raise ValueError('source must be 1 byte to 1 MiB')
        folder = self.root / 'candidate-text' / hashlib.sha256(data).hexdigest()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'source.py'
        if path.exists() and path.read_bytes() != data:
            raise ValueError('candidate text snapshot changed')
        if not path.exists():
            path.write_bytes(data)
            path.chmod(0o444)
        return self.artifact(path)

    def consume(self, gid, commit, *, deadline_monotonic):
        """Hold shared CAS flock only across lag check/update/checkpoint save.

        The external job supervisor MUST enforce this same finite deadline.
        No generation, measurement or native adapter call is allowed here.
        """
        if not time.monotonic() < deadline_monotonic <= time.monotonic() + 1800:
            raise ValueError('existing job deadline within 30 minutes required')
        # Raw proof/model-file checks precede the short consumption/CAS lock.
        # Recheck sealed attempt hashes after acquiring it. Same-user mutation
        # is outside the native cooperative-code containment guarantee.
        prepared = self.group(gid)
        if prepared['phase'] == 'complete':
            artifact(prepared['samples']['artifact'])
            for slot, item in enumerate(prepared['attempts']):
                attempt, checksum = self.checked_attempt(prepared, slot, item['attempt_id'])
                if checksum != item['attempt_sha256'] or attempt['score'] != prepared['sealed_rewards'][slot]:
                    raise ValueError('sealed group reward differs')
        with self.locked():
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError('job deadline passed before consumption lock acquired')
            group = self.group(gid)
            before, before_sha = self.latest_checkpoint(group['binding']['trainer'])
            if gid in before['consumed_group_ids'] + before['terminal_group_ids']:
                self.verify_committed_group(group, before)
                return dict(action='already_handled', group_id=gid, checkpoint_sha256=before_sha)
            if group['record_sha256'] != prepared['record_sha256']:
                raise ValueError('group changed during consumption preflight; retry from latest checkpoint')
            batch = self.prepare_batch(group, before_sha)
            group['prepared_batch'] = batch
            if batch['action'] == 'discard':
                group['phase'] = 'discarded'
                group['disposition'] = batch['reason']
            self.save_group(group)
            publication = None
            try:
                # Detached copy: callback cannot mutate native authorization.
                publication = commit(json.loads(json.dumps(batch)))
            finally:
                # Even when the callback raises after checkpoint publication,
                # persisted IDs remain authority; caller must reload that state.
                after, after_sha = self.latest_checkpoint(group['binding']['trainer'])
                if gid in after['consumed_group_ids'] + after['terminal_group_ids']:
                    self.verify_checkpoint_transition(group, batch, before, after)
                    if publication is None and sys.exc_info()[0] is None:
                        raise ValueError('callback must return committed checkpoint publication')
                    if publication is not None:
                        latest_path = Path(read(group['binding']['trainer']['checkpoint_pointer'])['checkpoint_path'])
                        expected = dict(checkpoint_path=str(latest_path), checkpoint_sha256=after_sha, group_id=gid,
                            disposition='consumed' if batch['action'] == 'update' else 'terminal',
                            policy_state_path=after['final_weights']['path'], policy_state_sha256=after['final_weights']['sha256'])
                        if publication != expected:
                            raise ValueError('callback publication does not name latest durable checkpoint/policy')
                    group['checkpoint_projection'] = dict(checkpoint_sha256=after_sha,
                        action=batch['action'], batch_sha256=batch['batch_sha256'])
                    self.save_group(group)
                elif sys.exc_info()[0] is None:
                    raise ValueError('trainer did not persist group outcome in latest checkpoint')
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError('consumption exceeded the externally supervised deadline')
            return dict(action=batch['action'], group_id=gid, checkpoint_sha256=after_sha,
                        batch_sha256=batch['batch_sha256'])

    def prepare_batch(self, group, before_sha):
        gid = group['group_id']
        current = read(self.root / 'champions' / (group['lane_id'] + '.json'))
        lag = current['revision'] - group['incumbent']['revision']
        if lag < 0 or (lag == 0 and current != group['incumbent']):
            raise ValueError('champion lineage regressed or changed at same revision')
        reason = group['disposition'] if group['phase'] == 'discarded' else None
        if reason is None and group['phase'] != 'complete':
            reason = 'incomplete_after_crash'
        if reason is None:
            if not group['samples']:
                raise ValueError('complete group lacks sealed sampled policy/source file')
            for slot, item in enumerate(group['attempts']):
                folder = self.root / 'rounds' / group['round_ids'][slot] / 'attempts' / item['attempt_id']
                if sha(folder / 'attempt.json') != item['attempt_sha256']:
                    raise ValueError('sealed attempt changed after consumption preflight')
        if reason is None and lag >= 2:
            reason = 'incumbent_revision_lag_at_least_two'
        counted = [(i, x) for i, x in enumerate(group['sealed_rewards']) if x is not None]
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for _, x in counted):
            raise ValueError('sealed reward must be null or finite in [0,1]')
        mean = math.fsum(x / len(counted) for _, x in counted) if counted else None
        std = math.sqrt(math.fsum((x - mean)**2 / len(counted) for _, x in counted)) if counted else None
        if reason is None and len(counted) < 2:
            reason = 'fewer_than_two_counted_samples'
        if reason is None and std == 0:
            reason = 'zero_variance'
        batch = dict(schema=CONTRACT, group_id=gid, action='discard' if reason else 'update', reason=reason,
            sealed_group_sha256=group['sealed_group_sha256'], binding_sha256=group['binding_sha256'],
            checkpoint_before_sha256=before_sha, incumbent_revision=group['incumbent']['revision'],
            incumbent_sha256=group['incumbent']['source_sha256'], current_revision=current['revision'],
            current_incumbent_sha256=current['source_sha256'],
            sealed_attempt_hashes=[item['attempt_sha256'] if item else None for item in group['attempts']],
            sealed_exclusions=group['sealed_exclusions'], samples=group['samples'],
            reset_checkpoint_sha256=group['binding']['rollout_checkpoint_sha256'],
            bindings=group['binding'],
            revision_lag=lag, stale_incumbent=lag == 1, sealed_rewards=group['sealed_rewards'],
            counted_indices=[] if reason else [i for i, _ in counted],
            counted_rewards=[] if reason else [x for _, x in counted], mean=mean, std=std,
            advantages=[] if reason else [(x - mean) / std for _, x in counted])
        batch['batch_sha256'] = digest(batch)
        return batch

    @staticmethod
    def verify_checkpoint_transition(group, batch, before, after):
        gid = group['group_id']
        if batch['batch_sha256'] != digest({k: v for k, v in batch.items() if k != 'batch_sha256'}):
            raise ValueError('native batch authorization changed')
        for key in ('consumed_group_ids', 'terminal_group_ids'):
            if not set(before[key]) <= set(after[key]):
                raise ValueError('checkpoint dropped prior group outcomes')
        if any(after['group_outcomes'].get(k) != v for k, v in before['group_outcomes'].items()):
            raise ValueError('checkpoint changed prior group outcome')
        target = 'consumed_group_ids' if batch['action'] == 'update' else 'terminal_group_ids'
        other = 'terminal_group_ids' if target == 'consumed_group_ids' else 'consumed_group_ids'
        expected = dict(action=batch['action'], batch_sha256=batch['batch_sha256'],
            sealed_group_sha256=batch['sealed_group_sha256'], counted_indices=batch['counted_indices'],
            consumed_rewards=batch['counted_rewards'], reason=batch['reason'])
        if (gid not in after[target] or gid in after[other] or after['group_outcomes'].get(gid) != expected
                or after.get('parent_checkpoint_sha256') != batch['checkpoint_before_sha256']
                or after['base_weights'] != before['base_weights']
                or after.get('parent_state_sha256') != before['final_weights']['sha256']
                or after['tensor_hashes']['base'] != before['tensor_hashes']['base']
                or after['tensor_hashes']['adapter_before'] != before['tensor_hashes']['adapter_after']):
            raise ValueError('checkpoint group/reward/parent/base binding differs')
        update = batch['action'] == 'update'
        if (after['step'] != before['step'] + int(update)
                or after['checkpoint_index'] != before['checkpoint_index'] + 1
                or after.get('optimizer_step_completed') is not update):
            raise ValueError('optimizer step proof differs from update/discard disposition')
        if not update and (after['tensor_hashes']['adapter_after'] != before['tensor_hashes']['adapter_after']
                           or after['optimizer_state_sha256'] != before['optimizer_state_sha256']):
            raise ValueError('discard must not update policy tensors or Adam state')
        if update and any(type(after.get('losses', {}).get(k)) not in (int, float)
                          or not math.isfinite(after['losses'][k]) for k in ('before', 'after')):
            raise ValueError('actual finite trainer losses required')

    def recover_groups(self):
        result = dict(reusable_complete=[], terminal_checkpoint_pending=[], already_handled=[])
        with self.locked():
            for path in sorted((self.root / 'groups').glob('*.json')):
                group = self.group(path.stem)
                checkpoint, _ = self.latest_checkpoint(group['binding']['trainer'])
                if group['group_id'] in checkpoint['consumed_group_ids'] + checkpoint['terminal_group_ids']:
                    self.verify_committed_group(group, checkpoint)
                    result['already_handled'].append(group['group_id'])
                elif group['phase'] == 'complete':
                    result['reusable_complete'].append(group['group_id'])
                else:
                    group['phase'] = 'discarded'
                    group['disposition'] = group['disposition'] or 'incomplete_after_crash'
                    self.save_group(group)
                    result['terminal_checkpoint_pending'].append(group['group_id'])
        return result

    def promote(self, gid, slot):
        if type(slot) is not int or not 0 <= slot < 8:
            raise ValueError('invalid group slot')
        group = self.group(gid)
        checkpoint, _ = self.latest_checkpoint(group['binding']['trainer'])
        if group['phase'] == 'discarded' or gid not in checkpoint['consumed_group_ids']:
            raise ValueError('dropped/unconsumed groups cannot promote')
        self.verify_committed_group(group, checkpoint)
        item = group['attempts'][slot]
        self.checked_attempt(group, slot, item['attempt_id'])
        return super().submit(group['round_ids'][slot], item['attempt_id'])

    def submit(self, rid, aid):
        raise ValueError('use promote(group_id, slot); direct submit bypasses consumption guard')


def aa_identity_fields(value):
    from group_adapter import aa_identity
    return aa_identity(value)
