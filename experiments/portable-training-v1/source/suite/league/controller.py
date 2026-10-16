"""Trusted-local league state. Filesystem ownership is the trust boundary, not a sandbox."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def relative_score(a, b):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (a,b)):
        raise ValueError('metrics must be finite nonnegative numbers')
    return 0.5 if a == b == 0 else (a / max(a,b)) / (a / max(a,b) + b / max(a,b))


def source_info(native):
    if 'incumbent' in native:
        return native['incumbent']['path'], native['incumbent']['sha256']
    if isinstance(native.get('incumbent_source'), dict):
        return native['incumbent_source']['path'], native['incumbent_source']['sha256']
    return native['incumbent_source'], native.get('incumbent_sha256', native.get('incumbent_source_sha256'))


def source_bind(native, path, checksum):
    if 'incumbent' in native:
        native['incumbent'] = dict(native['incumbent'], path=str(path), sha256=checksum)
    elif isinstance(native.get('incumbent_source'), dict):
        native['incumbent_source'] = dict(path=str(path), sha256=checksum)
    else:
        native['incumbent_source'] = str(path)
        if 'incumbent_sha256' in native: native['incumbent_sha256'] = checksum
        else: native['incumbent_source_sha256'] = checksum


def validate_lane(native, files, adapter):
    """Completeness and dependency closure; actual live checks belong to adapter."""
    env=native['environment']
    required={
        'kernel':('workload','runtime','runtime_sha256','scorer_sha256','oracle_sha256','budget_sha256','seed'),
        'inference':('runtime','bindings','inputs'),
        'training':('runtime','runtime_sha256','suite_sha256','data_manifest_sha256','model_identity','oracle_sha256','seeds','validation_seeds','validation_offset_tokens','data_dir'),
    }[env]
    missing=[k for k in required if k not in native or native[k] is None]
    if missing: raise ValueError('incomplete lane bindings: '+','.join(missing))
    if not isinstance(native['runtime'],dict) or not native['runtime']: raise ValueError('missing hardware/runtime descriptor')
    if env=='kernel':
        if not isinstance(native['scorer_sha256'],dict) or not native['scorer_sha256']: raise ValueError('missing evaluator closure')
        for path,checksum in native['scorer_sha256'].items():
            if files.get(str(Path(path).resolve()))!=checksum: raise ValueError('unpinned native dependency: '+path)
        if not native['workload'].get('cases') or not native['workload'].get('oracle'): raise ValueError('missing fixed oracle/validation cases')
    if env=='inference':
        for key in ('model_sha256','assets_manifest_sha256','data_sha256','oracle_sha256','evaluator_sha256','budget_sha256','runtime_sha256'):
            if not native['bindings'].get(key): raise ValueError('missing inference binding '+key)
        inputs=native['inputs']
        if inputs.get('train_partition')!='train' or inputs.get('validation_partition')!='train' or not inputs.get('train_nonce') or not inputs.get('validation_nonce') or inputs['train_nonce']==inputs['validation_nonce']:
            raise ValueError('independent train inputs required; final heldout excluded')
        dependencies=adapter.get('dependency_manifest',{})
        if not dependencies: raise ValueError('inference dependency_manifest required')
        hashes={name:files.get(str(Path(path).resolve())) for name,path in dependencies.items()}
        if any(v is None for v in hashes.values()) or digest(hashes)!=native['bindings']['evaluator_sha256']: raise ValueError('inference evaluator closure mismatch')
        cwd=Path(adapter['cwd'])
        for field,name in (('model_sha256','model.json'),('assets_manifest_sha256','assets-manifest.json'),('data_sha256','tasks.json'),('oracle_sha256','reference.py')):
            if files.get(str((cwd/name).resolve()))!=native['bindings'][field]: raise ValueError('unpinned inference asset '+name)
    if env=='training':
        if not native['seeds'] or not native['validation_seeds'] or set(native['seeds']) & set(native['validation_seeds']): raise ValueError('independent validation seeds required')
        for field,root in (('runtime_sha256',Path(adapter['cwd'])),('suite_sha256',Path(adapter['cwd']).parent/'suite')):
            if not isinstance(native[field],dict) or not native[field]: raise ValueError('missing training evaluator closure')
            for name,checksum in native[field].items():
                if files.get(str((root/name).resolve()))!=checksum: raise ValueError('unpinned training dependency '+name)
        if files.get(str((Path(native['data_dir'])/'manifest.json').resolve()))!=native['data_manifest_sha256']: raise ValueError('unpinned training data manifest')


def process_snapshot():
    rows = subprocess.check_output(['ps','-axo','pid=,ppid=,lstart=,command='], text=True, timeout=2)
    boot_file = Path('/proc/sys/kernel/random/boot_id')
    boot = boot_file.read_text().strip() if boot_file.exists() else None
    found = {}
    for line in rows.splitlines():
        fields=line.split(None,7)
        if len(fields) >= 7:
            pid = int(fields[0])
            # argv can change across exec without changing process ownership.
            # Linux start ticks also distinguish PID reuse within one second.
            if boot is not None:
                try:
                    stat = Path(f'/proc/{pid}/stat').read_text()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                ticks = stat.rsplit(')', 1)[1].split()[19]
                birth = f'linux:{boot}:{int(ticks)}'
            else:
                birth = 'ps-start:' + ' '.join(fields[2:7])
            found[pid] = dict(ppid=int(fields[1]), birth=birth,
                              command=fields[7] if len(fields) > 7 else '')
    return found


def observe_descendants(root, identities, rows):
    # Reparented descendants retain ownership only with matching birth identity.
    owned={pid for pid,birth in identities.items() if pid in rows and rows[pid]['birth']==birth}
    if root in rows and root not in identities: owned.add(root)
    while True:
        more={pid for pid,row in rows.items() if row['ppid'] in owned}
        if more<=owned: break
        owned |= more
    for pid in owned: identities[pid]=rows[pid]['birth']
    return owned


def run_bounded(argv, cwd, timeout, log):
    """Bounded identity-checked cleanup; same trusted-local boundary as adapters."""
    with Path(log).open('wb') as stream:
        proc = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        identities={}; deadline=time.monotonic()+timeout; expired=False
        try:
            while proc.poll() is None:
                observe_descendants(proc.pid,identities,process_snapshot())
                if time.monotonic()>=deadline:
                    expired=True; break
                time.sleep(.025)
        finally:
            try:
                rows=process_snapshot()
                owned=observe_descendants(proc.pid,identities,rows)
            except (subprocess.SubprocessError,OSError,ValueError):
                # Without refreshed identities, never signal remembered bare PIDs.
                owned=set()
            for pid in owned-{proc.pid}:
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
            # Live Popen child cannot have a recycled PID before reaping.
            if proc.poll() is None:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            proc.wait(timeout=2)
        return 'timeout' if expired else proc.returncode


class League:
    supported_environments = ('kernel', 'inference', 'training')

    @staticmethod
    def validate_native(native, files, adapter):
        if native['environment'] not in League.supported_environments:
            raise ValueError('unsupported lane; matching registration required')
        validate_lane(native, files, adapter)

    @staticmethod
    def lane_identity(native):
        return digest({k:native[k] for k in ('environment','workload_id','objective_name','profile','device','budget','runtime','runtime_sha256','bindings','data_manifest_sha256','suite_sha256','oracle_sha256') if k in native})

    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self):
        with (self.root / '.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(stream, fcntl.LOCK_UN)

    def event(self, value):
        # Called only while holding state lock. Never rewrite attempt/history events.
        with (self.root/'history.jsonl').open('ab') as stream:
            stream.write(canonical(dict(value, time_ns=time.time_ns())) + b'\n'); stream.flush(); os.fsync(stream.fileno())

    def artifact(self, source):
        data = Path(source).read_bytes()
        if not data or len(data) > 1_048_576: raise ValueError('source must be 1 byte to 1 MiB')
        checksum = hashlib.sha256(data).hexdigest(); path = self.root/'artifacts'/checksum/'source.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha(path) != checksum: raise ValueError('artifact mutated')
        else:
            path.write_bytes(data); path.chmod(0o444)
        return path, checksum

    def freeze(self, native_round, adapter, *, max_actions=10, timeout_seconds=600, synthetic=False):
        """Operator registers an environment-owned executable and its source files.

        adapter={argv:[python,script,'{round}','{candidate}','{output}'], cwd:path,
        trusted_files:[all evaluator source files]}. Candidate callers cannot set this.
        """
        native = json.loads(canonical(native_round))
        if type(max_actions) is not int or not 1 <= max_actions <= 100: raise ValueError('bounded max_actions required')
        if isinstance(timeout_seconds,bool) or not 0 < timeout_seconds <= 43200: raise ValueError('bounded timeout required')
        for key in ('environment','workload_id','objective_name','profile','device','budget','round_id'):
            if key not in native: raise ValueError('missing native round '+key)
        if native['environment'] not in self.supported_environments or native.get('lower_is_better',native['environment']=='training') is not True: raise ValueError('unsupported lane')
        origin, checksum = source_info(native)
        if sha(origin) != checksum: raise ValueError('incumbent identity mismatch')
        files = {str(Path(p).resolve()):sha(p) for p in adapter['trusted_files']}
        if not files or not adapter['argv'] or not all(isinstance(x,str) for x in adapter['argv']): raise ValueError('trusted adapter files and argv required')
        if not all(any('{'+x+'}' in arg for arg in adapter['argv']) for x in ('round','candidate','output')): raise ValueError('adapter placeholders required')
        self.validate_native(native,files,adapter)
        lane = self.lane_identity(native)
        with self.locked():
            artifact, checksum = self.artifact(origin)
            source_bind(native, artifact, checksum)
            champion_path = self.root/'champions'/f'{lane}.json'
            if champion_path.exists():
                champion = read(champion_path)
                if champion['source_sha256'] != checksum: raise ValueError('round must use current lane champion')
            else:
                champion = dict(revision=0, source_path=str(artifact), source_sha256=checksum, lane_id=lane)
                write(champion_path, champion)
            rid = uuid.uuid4().hex
            directory = self.root/'rounds'/rid; directory.mkdir(parents=True)
            write(directory/'native.json',native)
            round_value = dict(schema_version=1,round_id=rid,lane_id=lane,native_round=native,
                native_file_sha256=sha(directory/'native.json'),champion=champion,
                adapter=dict(argv=adapter['argv'],cwd=str(Path(adapter['cwd']).resolve()),trusted_files=files,dependency_manifest=adapter.get('dependency_manifest',{})),
                max_actions=max_actions,timeout_seconds=timeout_seconds,synthetic=bool(synthetic))
            write(directory/'round.json',round_value)
            write(directory/'seal.json',dict(round_sha256=sha(directory/'round.json')))
            self.event(dict(event='round_frozen',round_id=rid,lane_id=lane))
            return round_value

    def round(self, rid):
        if not isinstance(rid,str) or len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid): raise ValueError('invalid round id')
        directory=self.root/'rounds'/rid
        if sha(directory/'round.json') != read(directory/'seal.json')['round_sha256']: raise ValueError('round mutated')
        value=read(directory/'round.json')
        if sha(directory/'native.json') != value['native_file_sha256']: raise ValueError('native round mutated')
        if sha(value['champion']['source_path']) != value['champion']['source_sha256']: raise ValueError('incumbent mutated')
        self.validate_native(value['native_round'],value['adapter']['trusted_files'],value['adapter'])
        return value

    def execute(self, rid, candidate):
        with self.locked():
            r=self.round(rid)
            attempts=self.root/'rounds'/rid/'attempts'; attempts.mkdir(exist_ok=True)
            if len(list(attempts.iterdir())) >= r['max_actions']: raise ValueError('episode horizon exhausted')
            artifact, checksum=self.artifact(candidate)
            aid=uuid.uuid4().hex; folder=attempts/aid; folder.mkdir()
            pending=dict(attempt_id=aid,round_id=rid,candidate_path=str(artifact),candidate_sha256=checksum,status='running')
            write(folder/'attempt.json',pending); self.event(dict(event='attempt_started',round_id=rid,attempt_id=aid))
        result=dict(status='error',score=None,exclusion_reasons=[])
        try:
            self.check_sources(r)
            substitutions=dict(round=str(self.root/'rounds'/rid/'native.json'),candidate=str(artifact),output=str(folder/'output'))
            argv=[arg.format(**substitutions) for arg in r['adapter']['argv']]
            code=run_bounded(argv,r['adapter']['cwd'],r['timeout_seconds'],folder/'adapter.log')
            self.check_sources(r); self.round(rid)
            if sha(artifact)!=checksum: raise ValueError('candidate mutated')
            if code != 0: raise ValueError('adapter execution '+str(code))
            receipt=read(folder/'output'/'comparison.json')
            result=self.evaluate(r,checksum,receipt,folder/'output')
            proof={str(p.relative_to(folder)):sha(p) for p in (folder/'output').rglob('*') if p.is_file()}
            result.update(receipt=receipt,proof=proof)
        except Exception as exc:
            result=dict(status='blocked',score=None,promotion_eligible=False,exclusion_reasons=[str(exc)])
        with self.locked():
            final=dict(pending,**result); write(folder/'attempt.json',final)
            write(folder/'seal.json',dict(attempt_sha256=sha(folder/'attempt.json')))
            self.event(dict(event='attempt_finished',round_id=rid,attempt_id=aid,status=final['status'],score=final['score']))
        return final

    @staticmethod
    def check_sources(r):
        for path,checksum in r['adapter']['trusted_files'].items():
            if sha(path)!=checksum: raise ValueError('trusted adapter source mutated')

    @staticmethod
    def evaluate(r, candidate_sha, receipt, output):
        native=r['native_round']
        if native['environment'] not in League.supported_environments:
            raise ValueError('unsupported lane; matching registration required')
        for field in ('schema_version','environment','round_id','workload_id','objective_name','profile','device','budget'):
            if receipt.get(field)!=native.get(field): raise ValueError('receipt '+field+' mismatch')
        if receipt.get('lower_is_better') is not True: raise ValueError('objective direction mismatch')
        if receipt.get('round_sha256') not in (r['native_file_sha256'],digest(native)): raise ValueError('receipt round digest mismatch')
        for field in ('bindings','runtime_sha256','budget_sha256','oracle_sha256','data_manifest_sha256','suite_sha256','seeds','validation_seeds','inputs','runtime','model_identity','validation_offset_tokens'):
            if field in native and receipt.get(field)!=native[field]: raise ValueError('receipt binding mismatch: '+field)
        if receipt.get('incumbent_source_sha256')!=r['champion']['source_sha256'] or receipt.get('challenger_source_sha256')!=candidate_sha: raise ValueError('source receipt mismatch')
        for item in receipt.get('native_receipts',[]):
            path=Path(item['path']); path=path if path.is_absolute() else output/path
            path=path.resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file() or sha(path)!=item['sha256']: raise ValueError('native proof missing/changed/outside output')
        reasons=list(receipt.get('exclusion_reasons',[]))
        if r['synthetic'] or receipt.get('synthetic',False): reasons.append('synthetic fixture')
        if native['profile']!='standard': reasons.append('excluded profile')
        status=receipt.get('status')
        if status not in ('ok','invalid','blocked','error'): raise ValueError('unknown receipt status')
        if status=='invalid':
            excluded=(r['synthetic'] or receipt.get('synthetic',False) or native['profile']!='standard' or any(not str(reason).lower().startswith(('invalid','candidate','syntax')) for reason in reasons))
            return dict(status=status,score=None if excluded else 0.0,exclusion_reasons=reasons,promotion_eligible=False)
        if status!='ok': return dict(status=status,score=None,exclusion_reasons=reasons,promotion_eligible=False)
        if receipt.get('correctness_pass') is not True or receipt.get('comparison_eligible') is not True: reasons.append('comparison ineligible')
        if not receipt.get('native_receipts'): reasons.append('missing native proof')
        score=None if reasons else relative_score(receipt.get('incumbent_metric'),receipt.get('challenger_metric'))
        # Each environment has one authoritative gate. Optional generic flags
        # cannot override an explicit negative native lane verdict.
        env=native['environment']
        if env=='kernel':
            validation=receipt.get('validation_pass') is True
            improvement=receipt.get('promotion_eligible') is True
        elif env=='inference':
            validation=receipt.get('validation_pass') is True
            improvement=receipt.get('promotion_eligible') is True if 'promotion_eligible' in receipt else receipt.get('meaningful_improvement') is True
        else:
            validation=receipt.get('validation',{}).get('passed') is True
            improvement=receipt.get('promotion_ready') is True
        contradictory=(receipt.get('independent_validation',{}).get('passed') is False or receipt.get('meaningful_improvement') is False)
        promote=score is not None and score>.5 and validation and improvement and not contradictory
        return dict(status=status,score=score,exclusion_reasons=reasons,promotion_eligible=promote)

    def recover_committed(self):
        """Before another CAS, persist any current champion's missing terminal result."""
        for path in (self.root/'champions').glob('*.json'):
            champion=read(path)
            rid=champion.get('round_id'); aid=champion.get('attempt_id')
            if not rid or not aid: continue
            folder=self.root/'rounds'/rid/'attempts'/aid
            if (folder/'submitted.json').exists(): continue
            self.round(rid)
            if sha(folder/'attempt.json')!=read(folder/'seal.json')['attempt_sha256']: raise ValueError('committed attempt mutated')
            attempt=read(folder/'attempt.json')
            if attempt['candidate_sha256']!=champion['source_sha256']: raise ValueError('committed champion identity mismatch')
            result=dict(attempt_id=aid,round_id=rid,score=attempt['score'],promoted=True,stale=False)
            write(folder/'submitted.json',result)
            self.event(dict(event='submitted_recovered',**result))

    def submit(self,rid,aid):
        if not isinstance(aid,str) or len(aid)!=32 or any(c not in '0123456789abcdef' for c in aid): raise ValueError('invalid attempt id')
        with self.locked():
            r=self.round(rid); self.check_sources(r)
            self.recover_committed()
            folder=self.root/'rounds'/rid/'attempts'/aid
            if sha(folder/'attempt.json')!=read(folder/'seal.json')['attempt_sha256']: raise ValueError('attempt mutated')
            attempt=read(folder/'attempt.json')
            if (folder/'submitted.json').exists(): return read(folder/'submitted.json')
            if sha(attempt['candidate_path'])!=attempt['candidate_sha256']: raise ValueError('candidate mutated')
            for path,checksum in attempt.get('proof',{}).items():
                if sha(folder/path)!=checksum: raise ValueError('proof mutated')
            if 'receipt' in attempt:
                checked=self.evaluate(r,attempt['candidate_sha256'],attempt['receipt'],folder/'output')
                if any(checked[k]!=attempt.get(k) for k in ('status','score','promotion_eligible')):
                    raise ValueError('terminal result differs from sealed native proof')
                eligible=checked['promotion_eligible'] is True
            else:
                eligible=False
            champion_path=self.root/'champions'/f"{r['lane_id']}.json"; current=read(champion_path)
            recovered=current.get('attempt_id')==aid and current.get('source_sha256')==attempt['candidate_sha256']
            stale=current!=r['champion'] and not recovered
            promoted=eligible and (not stale or recovered)
            result=dict(attempt_id=aid,round_id=rid,score=attempt['score'],promoted=promoted,stale=stale)
            if promoted and not recovered:
                write(champion_path,dict(revision=current['revision']+1,lane_id=r['lane_id'],source_path=attempt['candidate_path'],source_sha256=attempt['candidate_sha256'],attempt_id=aid,round_id=rid))
            write(folder/'submitted.json',result)
            self.event(dict(event='submitted',**result)); return result

    def status(self):
        return dict(champions=[read(p) for p in sorted((self.root/'champions').glob('*.json'))],rounds=[p.name for p in (self.root/'rounds').glob('*')],live_execution_claim=False)
