"""Independent scorer. Treat submissions as trusted local code, not a sandbox."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import signal
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import torch
from assets import HERE, local_model, sha
from shared_contract import accelerator_lock, normalize, speed_reward
from cache_probes import inputs as cache_inputs

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ATOL = RTOL = 0.002
LENGTHS = (32, 128, 512)
VOCAB_SIZE = 49152
LIMITS = {'development_calls': 10, 'final_calls': 1, 'worker_wall_s': 180,
          'evaluation_wall_s': 600, 'worker_rss_bytes': 6_000_000_000,
          'mps_driver_bytes': 4_000_000_000, 'cpu_threads': 2}
CONTROLLER_HASHES = {name: sha(HERE/name) for name in ('scorer.py','assets.py','shared_contract.py','tasks.json','controller.py','cache_probes.py','supervisor.py')}

class CandidateFailure(RuntimeError):
    pass

class OracleFailure(RuntimeError):
    """The evaluator cannot supply a consistent correctness reference."""
    pass

def validate_artifact(artifact, task):
    """Bind worker coverage to saved inputs, independently of peer agreement.

    Agreement between two incomplete outputs is not a correctness oracle.
    Legacy comparison-only callers may omit task; executing V2 scoring cannot.
    """
    rows = artifact['rows']
    cases = task['cases']
    if not cases or len(rows) != len(cases):
        raise ValueError('artifact does not cover the input cases')
    vocabulary = task.get('vocab_size', VOCAB_SIZE)
    steps = task['decode_steps']
    if len({case['key'] for case in cases}) != len(cases):
        raise ValueError('duplicate input case identities')

    def logits(value, shape, label):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f'{label}: expected shape {tuple(shape)}')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'{label}: expected finite floating logits')

    for row, case in zip(rows, cases):
        batch, length = case['ids'].shape
        if row['key'] != case['key'] or row['length'] != length:
            raise ValueError('artifact case identity/length differs from inputs')
        logits(row['first'], (batch, vocabulary), 'first')
        logits(row['greedy'], (steps, batch, vocabulary), 'greedy')
        logits(row['forced'], (len(case['forced']), batch, vocabulary), 'forced')
        tokens = row['tokens']
        if (not isinstance(tokens, torch.Tensor) or tokens.shape != (steps, batch, 1)
                or tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError('generated tokens: wrong shape or integer dtype')
        selected = torch.cat([row['first'].unsqueeze(0), row['greedy'][:-1]], dim=0).argmax(-1, keepdim=True)
        if not torch.equal(tokens, selected):
            raise ValueError('generated tokens do not follow recorded greedy logits')

    if 'cache_probe_inputs' not in task:
        raise ValueError('V2 task is missing required cache probe inputs')
    probes = task['cache_probe_inputs']
    a, b, branches = probes['a'], probes['b'], probes['branches']
    # Deliberately specify expected contexts outside the worker's probe runner.
    expected = [
        ('request-a', a), ('request-a-step', torch.cat([a, branches[0]], 1)),
        ('request-b', b), ('request-a-after-b', a),
        ('branch-left-1', torch.cat([a, branches[0]], 1)),
        ('branch-left-2', torch.cat([a, branches[0], branches[1]], 1)),
        ('branch-right-reset', a), ('branch-right-1', torch.cat([a, branches[2]], 1)),
        ('repeated-prefix-extension', torch.cat([a, probes['extension']], 1)),
        ('shorter-prefix-reset', a[:, :3]), ('batch-two-reset', probes['batch']),
        ('batch-two-step', torch.cat([probes['batch'], probes['batch_tokens']], 1)),
        ('batch-one-after-two', a),
    ]
    actual = artifact.get('cache_probes', [])
    if len(actual) != len(expected):
        raise ValueError('V2 artifact must contain all thirteen cache probes')
    for row, (key, context) in zip(actual, expected):
        if row['key'] != key or not torch.equal(row['context'], context):
            raise ValueError('cache probe identity/context differs from inputs')
        logits(row['logits'], (context.shape[0], vocabulary), f'cache probe {key}')

def compare(reference, candidate, task=None):
    metrics = {'atol': ATOL, 'rtol': RTOL}
    try:
        if task is not None:
            validate_artifact(reference, task)
            validate_artifact(candidate, task)
        expected, actual = reference['rows'], candidate['rows']
        if len(expected) != len(actual) or not expected:
            raise ValueError('wrong number of cases')
        worst = 0.0
        for ref, got in zip(expected, actual):
            if ref['key'] != got['key'] or ref['length'] != got['length']:
                raise ValueError('case identity mismatch')
            for name in ('first', 'greedy', 'forced'):
                x, y = ref[name], got[name]
                if x.shape != y.shape or not torch.isfinite(y).all():
                    raise ValueError(f'{name}: shape or nonfinite output')
                worst = max(worst, float((x - y).abs().max()))
                metrics['max_abs_logit_error'] = worst
                if not torch.allclose(x, y, atol=ATOL, rtol=RTOL):
                    raise ValueError(f'{name}: logits exceed tolerance')
                if not torch.equal(x.argmax(-1), y.argmax(-1)):
                    raise ValueError(f'{name}: greedy token mismatch')
            if not torch.equal(ref['tokens'], got['tokens']):
                raise ValueError('generated token mismatch')
        metrics['max_abs_logit_error'] = worst
        expected_probes=reference.get('cache_probes',[])
        actual_probes=candidate.get('cache_probes',[])
        if len(expected_probes)!=len(actual_probes):
            raise ValueError('missing cache correctness probes')
        for ref,got in zip(expected_probes,actual_probes):
            if ref['key']!=got['key'] or not torch.equal(ref['context'],got['context']):
                raise ValueError('cache probe identity/context mismatch')
            x,y=ref['logits'],got['logits']
            if x.shape!=y.shape or not torch.isfinite(y).all() or not torch.allclose(x,y,atol=ATOL,rtol=RTOL):
                raise ValueError(f'cache probe {ref["key"]}: logits fail correctness')
            if not torch.equal(x.argmax(-1),y.argmax(-1)):
                raise ValueError(f'cache probe {ref["key"]}: greedy token mismatch')
        metrics['cache_probes_checked']=len(expected_probes)
        return {'ok': True, 'reward': 0.0, 'metrics': metrics}
    except Exception as exc:
        return {'ok': False, 'reward': 0.0, 'metrics': metrics,
                'error': f'{type(exc).__name__}: {exc}'}

def machine(ignore_pids=()):
    processes = subprocess.check_output(['ps', '-Ao', 'pid,pcpu,rss,comm'], text=True)
    rows = []
    blockers = []
    own = {os.getpid(), os.getppid(), *ignore_pids}
    for line in processes.splitlines()[1:]:
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, cpu, rss, command = fields
        row = {'pid': int(pid), 'cpu_percent': float(cpu), 'rss_kb': int(rss), 'command': command}
        rows.append(row)
        if int(pid) in own:
            continue
        if any(x in command.lower() for x in ('leagueoflegends', 'league of legends', 'chrome-headless-shell', 'ffmpeg', 'blender')):
            blockers.append(row)
        elif float(cpu) >= 80:
            blockers.append(row)
    # Detect other known measurement jobs even when GPU dispatch has low CPU.
    args = subprocess.check_output(['ps', '-Ao', 'pid,args'], text=True)
    for line in args.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) not in own and any(x in parts[1] for x in ('kernels/bench.py', 'rsi/train.py', 'rsi/score.py', 'inference/worker.py')):
            blockers.append({'pid': int(parts[0]), 'command': 'another known RSI measurement worker'})
    return {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'platform': platform.platform(), 'python': sys.version, 'machine': platform.machine(),
        'hardware': subprocess.check_output(['sysctl', 'hw.model', 'hw.memsize', 'machdep.cpu.brand_string'], text=True).strip(),
        'versions': {p: importlib.metadata.version(p) for p in ('torch','transformers','huggingface-hub','safetensors','tokenizers','numpy')},
        'mps_available': torch.backends.mps.is_available(), 'blockers': blockers,
        'top_processes': sorted(rows, key=lambda x: -x['cpu_percent'])[:12]}

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        # Nonblocking: waiting is visible to research-loop callers, no hidden queue.
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def measurement_lock_path():
    target=Path(os.environ.get('RSI_INFERENCE_MEASUREMENT_LOCK','/tmp/rsi-inference-measurement.lock')).expanduser()
    if not target.is_absolute():raise ValueError('measurement lock override must be absolute')
    return target

def make_inputs(mode, nonce, partition=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model(), local_files_only=True)
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise ValueError('pinned model vocabulary differs from the evaluation contract')
    generator = torch.Generator().manual_seed(int(nonce[:15], 16))
    trials = 5 if mode == 'final' else (1 if mode == 'check' else 3)
    count = 2
    partition = partition or ('heldout' if mode == 'final' else 'train')
    spec = json.loads((HERE / 'tasks.json').read_text())[partition]
    themes = spec['themes']
    cases = []
    for trial in range(trials):
        for length in LENGTHS:
            for sample in range(count):
                if sample == 1:
                    ids = torch.randint(3, tokenizer.vocab_size, (1, length), generator=generator)
                else:
                    words = [themes[int(torch.randint(len(themes), (), generator=generator))] for _ in range(length)]
                    text = 'Explain this observation carefully. ' + ' '.join(f'The {w} changed by {int(torch.randint(10000, (), generator=generator))} units.' for w in words)
                    ids = tokenizer(text, return_tensors='pt')['input_ids'][:, :length]
                    if ids.shape[1] != length:
                        raise ValueError('prompt construction too short')
                forced = torch.randint(3, tokenizer.vocab_size, (2, 1, 1), generator=generator)
                cases.append(dict(key=f'{trial}:{length}:{sample}', ids=ids, forced=forced))
    return {'cases': cases, 'decode_steps': 4 if mode == 'check' else 8,
            'vocab_size': VOCAB_SIZE,
            'partition': partition, 'task_id': spec['task_id'],
            'cache_probe_inputs':cache_inputs(nonce,tokenizer.vocab_size)}

WORKER_FAILURE_STAGES = {
    'dependency_import', 'worker_setup', 'asset_lookup', 'model_load', 'input_load',
    'candidate_import', 'engine_init', 'prefill', 'decode', 'candidate_contract',
    'device_sync', 'reference_oracle', 'worker_processing', 'cache_probes', 'output_save',
}

def read_worker_failure(output, source, pid, returncode):
    """Only source/PID-bound direct candidate failures earn zero credit.

    No usable receipt, a signal, or unknown attribution is infrastructure. This
    validates bookkeeping for trusted code, not authentication of a hostile child.
    """
    path = output.with_suffix('.failure.json')
    try:
        with path.open('rb') as stream:
            content = stream.read(16385)
        if len(content) > 16384:
            raise ValueError('worker failure receipt exceeds 16 KiB')
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {
                'schema_version', 'pid', 'source_sha256', 'stage', 'classification', 'error'}:
            raise ValueError('worker failure receipt schema mismatch')
        if (type(value['schema_version']) is not int or value['schema_version'] != 1
                or type(value['pid']) is not int or value['pid'] != pid
                or value['source_sha256'] != sha(source)):
            raise ValueError('worker failure receipt identity mismatch')
        if (not isinstance(value['stage'], str) or value['stage'] not in WORKER_FAILURE_STAGES
                or not isinstance(value['classification'], str)
                or value['classification'] not in ('candidate', 'infrastructure', 'unknown')):
            raise ValueError('worker failure receipt attribution mismatch')
        error = value['error']
        if (not isinstance(error, dict) or set(error) != {'type', 'message', 'origin'}
                or not isinstance(error['type'], str) or not 1 <= len(error['type']) <= 128
                or not isinstance(error['message'], str) or len(error['message']) > 2048
                or error['origin'] not in ('submission', 'worker', 'provider', 'unknown', 'contract')):
            raise ValueError('worker failure receipt error schema mismatch')
        if value['classification'] == 'candidate':
            direct = value['stage'] in {'candidate_import', 'engine_init', 'prefill', 'decode'} and error['origin'] == 'submission'
            contract = value['stage'] == 'candidate_contract' and error['origin'] == 'contract'
            if not (direct or contract) or returncode <= 0:
                raise ValueError('unsupported candidate failure attribution')
        return value
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {'classification': 'unknown', 'receipt_error': f'{type(exc).__name__}: {exc}'}

def child(source, inputs, output, device, deadline):
    cmd = [sys.executable, str(output.parent / 'worker.py'), str(source), str(inputs), str(output), '--device', device]
    started = time.monotonic()
    peak = 0
    observations = []
    next_check = started
    failure = None
    failure_path = output.with_suffix('.failure.json')
    # Each invocation owns its fresh output path; never reuse an old failure.
    failure_path.unlink(missing_ok=True)
    with output.with_suffix('.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True,
            env=dict(os.environ, TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2'))
        try:
            while proc.poll() is None:
                if time.monotonic() - started > LIMITS['worker_wall_s'] or time.monotonic() > deadline:
                    raise TimeoutError('worker/evaluation wall budget exceeded')
                process_memory = subprocess.run(['ps', '-axo', 'pgid=,rss='], capture_output=True, text=True).stdout
                rss = sum(int(parts[1]) * 1024 for line in process_memory.splitlines()
                    if len(parts := line.split()) == 2 and int(parts[0]) == proc.pid)
                peak = max(peak, rss)
                if peak > LIMITS['worker_rss_bytes']:
                    raise MemoryError('worker RSS budget exceeded')
                if time.monotonic() >= next_check:
                    observations.append(machine(ignore_pids=(proc.pid,)))
                    next_check = time.monotonic() + 5
                time.sleep(0.25)
            if proc.returncode:
                failure = read_worker_failure(output, source, proc.pid, proc.returncode)
                message = f'worker exit {proc.returncode}; see {output.with_suffix(".log").name}'
                if failure is not None:
                    message += f'; {failure.get("stage", "unknown")}: {failure.get("error", {})}'
                if failure is not None and failure.get('classification') == 'candidate':
                    raise CandidateFailure(message)
                raise RuntimeError(message)
        finally:
            # Clean up only this scorer-created process group, including any
            # submission descendants. Never touch unrelated applications.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if proc.poll() is None:
                proc.wait()
            output.with_suffix('.worker.json').write_text(json.dumps({
                'wall_s': time.monotonic()-started, 'peak_sampled_rss_bytes': peak,
                'returncode': proc.returncode, 'conditions': observations,
                'failure_report': failure}, indent=2)+'\n')
    return {'wall_s': time.monotonic() - started, 'peak_sampled_rss_bytes': peak,
            'artifact_sha256': sha(output), 'conditions': observations}

def timing(reference, candidate, bookend, steps):
    segments = []
    drift = []
    noise = []
    for length in LENGTHS:
        a = [r for r in reference['rows'] if r['length'] == length]
        b = [r for r in candidate['rows'] if r['length'] == length]
        c = [r for r in bookend['rows'] if r['length'] == length]
        for phase in ('prefill', 'decode'):
            name = phase + '_s'
            x, y, z = [statistics.median(r[name] for r in group) for group in (a, b, c)]
            if not all(math.isfinite(v) and v > 0 for v in (x,y,z)):
                raise ValueError('invalid timing')
            baseline = (x + z) / 2
            drift.append(max(x/z, z/x) - 1)
            samples = [r[name] for r in a+c]
            center = statistics.median(samples)
            noise.append(statistics.median(abs(v-center) for v in samples)/center)
            segments.append(dict(length=length, phase=phase,
                baseline_ms=baseline*1000, candidate_ms=y*1000, speedup=baseline/y,
                baseline_samples_ms=[r[name]*1000 for r in a+c],
                candidate_samples_ms=[r[name]*1000 for r in b],
                candidate_tokens_per_s=(length if phase == 'prefill' else steps)/y))
    speedup = math.exp(statistics.mean(math.log(s['speedup']) for s in segments))
    return {'segments': segments, 'geomean_speedup': speedup,
            'max_bookend_drift': max(drift), 'max_baseline_relative_mad': max(noise),
            'noise_floor': max(0.05, max(drift), 2*max(noise))}

def _evaluate(solution, episode, mode='development', device='cpu', diagnostic=False, partition=None):
    if mode not in ('development','final','check') or device not in ('cpu','mps'):
        raise ValueError('unknown mode/device')
    partition = partition or ('heldout' if mode == 'final' else 'train')
    if partition not in ('train','heldout'):
        raise ValueError('unknown task partition')
    solution, episode = Path(solution).resolve(), Path(episode).resolve()
    episode.mkdir(parents=True, exist_ok=True)
    with lock(episode / '.episode.lock'), lock(measurement_lock_path()), accelerator_lock(device=device, timeout_seconds=0) as lease:
        state_path = episode / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'development': 0, 'final': 0}
        if state['final']:
            raise ValueError('episode sealed by final evaluation')
        bucket = 'final' if mode == 'final' else 'development'
        if state[bucket] >= (1 if bucket == 'final' else 10):
            raise ValueError('experiment call budget exhausted')
        state[bucket] += 1
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        nonce = secrets.token_hex(16)
        directory = episode / f'{mode}-{state[bucket]:02d}-{nonce[:8]}'
        directory.mkdir()
        shutil.copyfile(solution, directory / 'submission.py')
        for name in ('scorer.py','controller.py','supervisor.py','cache_probes.py','worker.py','reference.py','assets.py','model.json','assets-manifest.json','requirements.txt','requirements-lock.txt','shared_contract.py','tasks.json'):
            shutil.copyfile(HERE / name, directory / name)
        (directory / 'shared').mkdir()
        for function, name in ((accelerator_lock,'resources.py'), (speed_reward,'records.py')):
            shutil.copyfile(Path(sys.modules[function.__module__].__file__), directory / 'shared' / name)
        (directory / 'shared' / '__init__.py').touch()
        result = {'schema_version': 1, 'task': 'rsi.inference.smollm2-135m.v2',
            'mode': mode, 'device': device, 'nonce': nonce, 'limits': LIMITS,
            'ok': False, 'reward': None, 'timing_valid': False,
            'evidence_kind': 'not_executed', 'artifact_dir': str(directory),
            'source_sha256': sha(directory / 'submission.py'),
            'evaluator_hashes': {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()},
            'model': json.loads((HERE / 'model.json').read_text()), 'conditions': [], 'workers': {}}
        result['accelerator_lease'] = lease
        result['controller_hashes_at_import'] = CONTROLLER_HASHES
        result['partition'] = partition
        start = time.monotonic()
        try:
            local_model()
            before = machine()
            result['machine_before'] = before
            if before['blockers'] and not diagnostic:
                result['status'] = 'blocked_timing'
                raise RuntimeError('competing render/game/heavy CPU process; use diagnostic for correctness only')
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable')
            inputs = make_inputs(mode, nonce, partition)
            result['partition_task_id'] = inputs['task_id']
            torch.save(inputs, directory / 'inputs.pt')
            result['inputs_sha256'] = sha(directory / 'inputs.pt')
            for name, source in [('baseline_before', directory/'reference.py'),
                                 ('candidate', directory/'submission.py'),
                                 ('baseline_after', directory/'reference.py')]:
                result['conditions'].append(machine())
                try:
                    result['workers'][name] = child(source, directory/'inputs.pt', directory/f'{name}.pt', device, start+600)
                    result['evidence_kind'] = 'pretrained_model'
                except CandidateFailure as exc:
                    if name != 'candidate' or time.monotonic() >= start+600:
                        raise RuntimeError(f'{name} failed: {exc}') from exc
                    raise
            reference, candidate, bookend = [torch.load(directory/f'{name}.pt', weights_only=True) for name in ('baseline_before','candidate','baseline_after')]
            for name, value in [('baseline_before', reference), ('baseline_after', bookend)]:
                try:
                    validate_artifact(value, inputs)
                except Exception as exc:
                    raise OracleFailure(f'{name} artifact incomplete: {exc}') from exc
            verdict = compare(reference, candidate, inputs)
            result.update(verdict)
            baseline_verdict = compare(reference, bookend, inputs)
            result['metrics']['candidate_matches_initial_reference'] = verdict['ok']
            result['metrics']['baseline_comparison'] = baseline_verdict
            if not baseline_verdict['ok']:
                raise OracleFailure('baseline nondeterminism: '+baseline_verdict['error'])
            for name, digest in result['evaluator_hashes'].items():
                if sha(directory/name) != digest:
                    raise ValueError(f'snapshot mutated: {name}')
            for name, digest in CONTROLLER_HASHES.items():
                if sha(HERE/name) != digest or sha(directory/name) != digest:
                    raise ValueError(f'controller source changed since import: {name}')
            if sha(directory/'inputs.pt') != result['inputs_sha256']:
                raise ValueError('input artifact mutated')
            if not verdict['ok']:
                # An incomplete/wrong candidate need not have timing samples.
                # Validate references and integrity first, then preserve zero
                # credit instead of failing later in timing aggregation.
                raise CandidateFailure(verdict['error'])
            result['metrics'].update(timing(reference, candidate, bookend, inputs['decode_steps']))
            result['metrics']['setup'] = {name: {
                'engine_init_s': value.get('engine_init_s'), 'warmup_s': value['warmup_s'],
                'mps_memory': value['mps_memory'], 'torch_threads': value.get('torch_threads')}
                for name, value in [('baseline_before',reference), ('candidate',candidate), ('baseline_after',bookend)]}
            after = machine()
            result['machine_after'] = after
            observed = [m for w in result['workers'].values() for m in w['conditions']]
            clean = not any(m['blockers'] for m in [before,after]+result['conditions']+observed)
            stable = result['metrics']['max_bookend_drift'] <= 0.05 and result['metrics']['max_baseline_relative_mad'] <= 0.05
            result['timing_valid'] = clean and stable and not diagnostic and mode != 'check'
            result['status'] = ('passed' if result['ok'] else 'incorrect') if result['timing_valid'] else ('diagnostic_pass' if result['ok'] else 'diagnostic_incorrect')
            result['metrics']['effect_above_noise'] = result['metrics']['geomean_speedup'] > 1 + result['metrics']['noise_floor']
            if result['ok'] and result['timing_valid']:
                result['reward'] = speed_reward(result['metrics']['geomean_speedup'])
        except Exception as exc:
            result['ok'] = False
            if isinstance(exc, CandidateFailure):
                result['status'] = 'incorrect'
                result['reward'] = 0.0
            else:
                # Evaluator failures override an earlier candidate verdict. An
                # inconsistent oracle must never produce trainable zero credit.
                if result.get('status') != 'blocked_timing':
                    result['status'] = 'failed'
                result['reward'] = None
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['evaluation_wall_s'] = time.monotonic()-start
        result['record'] = normalize(result)
        (directory / 'record.json').write_text(json.dumps(result['record'], indent=2, allow_nan=False)+'\n')
        (directory / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        with (episode / 'feedback.jsonl').open('a') as f:
            f.write(json.dumps(result, allow_nan=False)+'\n')
        with (episode / 'records.jsonl').open('a') as f:
            f.write(json.dumps(result['record'], allow_nan=False)+'\n')
        return result

def _evaluate_with_errors(solution, episode, mode='development', device='cpu', diagnostic=False, partition=None):
    try:
        return _evaluate(solution, episode, mode, device, diagnostic, partition)
    except Exception as exc:
        # Lease contention, exhausted/sealed episodes, missing submissions and
        # setup failures still produce an exclusion record at the interface.
        nonce = secrets.token_hex(16)
        directory = Path(episode).resolve() / f'event-{nonce[:8]}'
        directory.mkdir(parents=True, exist_ok=True)
        busy = isinstance(exc, (BlockingIOError, TimeoutError))
        raw = {'schema_version': 1, 'task': 'rsi.inference.smollm2-135m.v2',
            'mode': mode, 'device': device, 'nonce': nonce, 'limits': LIMITS,
            'ok': False, 'reward': None, 'timing_valid': False,
            'status': 'blocked_timing' if busy else 'failed',
            'error': f'{type(exc).__name__}: {exc}', 'evaluation_wall_s': 0.0,
            'artifact_dir': str(directory), 'evidence_kind': 'not_executed',
            'source_sha256': sha(solution) if Path(solution).is_file() else None,
            'evaluator_hashes': CONTROLLER_HASHES,
            'model': json.loads((HERE / 'model.json').read_text())}
        raw['record'] = normalize(raw)
        (directory / 'result.json').write_text(json.dumps(raw, indent=2)+'\n')
        (directory / 'record.json').write_text(json.dumps(raw['record'], indent=2)+'\n')
        return raw

def controller_command(config,output):
    return [sys.executable,str(HERE/'controller.py'),str(config),str(output)]

def evaluate(solution, episode, mode='development', device='cpu', diagnostic=False, partition=None):
    """Bound the entire executing evaluator, including preflight and snapshots."""
    from supervisor import run
    nonce = secrets.token_hex(16)
    directory = Path(episode).resolve() / f'controller-{nonce[:8]}'
    directory.mkdir(parents=True,exist_ok=False)
    config = directory/'config.json'
    output = directory/'output.json'
    config.write_text(json.dumps({'solution':str(Path(solution).resolve()),
        'episode':str(Path(episode).resolve()), 'mode':mode, 'device':device,
        'diagnostic':diagnostic, 'partition':partition})+'\n')
    started = time.monotonic()
    try:
        supervision = run(controller_command(config,output),
                          output,LIMITS['evaluation_wall_s'],LIMITS['worker_rss_bytes'])
        if output.stat().st_size > 16_000_000:
            raise ValueError('controller JSON result exceeds 16 MB')
        result = json.loads(output.read_text())
        result['supervision'] = supervision
        result['supervised_wall_s'] = time.monotonic()-started
        # Controller writes the inner receipt; preserve a separate full-call
        # envelope rather than rewriting its artifact hashes or measured times.
        (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    except Exception as exc:
        result = {'schema_version':1, 'task':'rsi.inference.smollm2-135m.v2',
            'mode':mode,'device':device,'nonce':nonce,'limits':LIMITS,
            'ok':False,'reward':None,'timing_valid':False,'status':'failed',
            'error':f'{type(exc).__name__}: {exc}',
            'evaluation_wall_s':time.monotonic()-started,
            'artifact_dir':str(directory),'evidence_kind':'execution_unverified',
            'source_sha256':None,'evaluator_hashes':CONTROLLER_HASHES,
            'model':json.loads((HERE/'model.json').read_text()),
            'supervision':getattr(exc,'supervisor_metadata',{})}
        result['record'] = normalize(result)
        (directory/'record.json').write_text(json.dumps(result['record'],indent=2)+'\n')
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('solution', type=Path)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--mode', choices=['development','final','check'], default='development')
    p.add_argument('--device', choices=['cpu','mps'], default='cpu')
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--partition', choices=['train','heldout'])
    a = p.parse_args()
    print(json.dumps(evaluate(a.solution, a.episode, a.mode, a.device, a.diagnostic, a.partition)))
