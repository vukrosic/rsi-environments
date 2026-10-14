"""Evaluator-owned child. Baseline and candidate run in separate processes."""
import argparse
import dis
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import secrets
from pathlib import Path

FAILURE_STAGES = frozenset(('dependency_import', 'worker_setup', 'asset_lookup',
    'model_load', 'input_load', 'candidate_import', 'engine_init', 'prefill',
    'decode', 'candidate_contract', 'device_sync', 'reference_oracle',
    'worker_processing', 'cache_probes', 'output_save'))
_CANDIDATE_STAGES = frozenset(('candidate_import', 'engine_init', 'prefill', 'decode'))
_PROVIDERS = ('torch', 'transformers', 'huggingface_hub', 'safetensors', 'tokenizers')
torch = None
_CacheContractViolation = None


class _ContractViolation(ValueError):
    """An explicit evaluator-owned check of the submitted Engine contract."""


class _BoundaryFailure(RuntimeError):
    def __init__(self, stage, error):
        self.stage, self.error = stage, error
        super().__init__(f'worker failed during {stage}')


def _dependencies():
    # Deferred imports let startup dependency failures produce a receipt too.
    import torch
    from transformers import AutoModelForCausalLM
    from assets import local_model
    from cache_probes import ContractViolation, run as run_cache_probes
    global _CacheContractViolation
    _CacheContractViolation = ContractViolation
    return torch, AutoModelForCausalLM, local_model, run_cache_probes


def _call(stage, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except _BoundaryFailure:
        raise
    except BaseException as error:
        if (stage == 'cache_probes' and _CacheContractViolation is not None and
                isinstance(error, _CacheContractViolation)):
            error = _ContractViolation(str(error))
        if isinstance(error, _ContractViolation):
            stage = 'candidate_contract'
        raise _BoundaryFailure(stage, error) from error


def _provider(module):
    return any(module == name or module.startswith(name + '.') for name in _PROVIDERS)


def _explicit_raise(traceback):
    instruction = None
    for item in dis.get_instructions(traceback.tb_frame.f_code):
        if item.offset > traceback.tb_lasti:
            break
        instruction = item.opname
    return instruction == 'RAISE_VARARGS'


def _failure_attribution(error, stage, source):
    """Use observed origin, never stderr wording, to attribute a failure."""
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    filename = traceback.tb_frame.f_code.co_filename if traceback is not None else None
    module = traceback.tb_frame.f_globals.get('__name__', '') if traceback is not None else ''
    direct = filename is not None and Path(filename).resolve() == source.resolve()
    syntax = isinstance(error, SyntaxError) and error.filename is not None and Path(error.filename).resolve() == source.resolve()
    if isinstance(error, _ContractViolation):
        origin = 'contract'
    elif direct or syntax:
        origin = 'submission'
    elif _provider(module) or _provider(type(error).__module__):
        origin = 'provider'
    elif stage not in _CANDIDATE_STAGES:
        origin = 'worker'
    else:
        origin = 'unknown'
    if source.name == 'reference.py':
        return 'infrastructure', origin
    if isinstance(error, _ContractViolation):
        return 'candidate', 'contract'
    if stage not in _CANDIDATE_STAGES:
        return 'infrastructure', origin
    seen = set()
    pending = [(error, frozenset())]
    resource_or_provider = origin == 'provider'
    unknown_chain = False
    while pending and len(seen) < 16:
        causal, ancestors = pending.pop()
        if id(causal) in ancestors:
            unknown_chain = True
            continue
        if id(causal) in seen:
            continue
        seen.add(id(causal))
        resource_or_provider |= (isinstance(causal, (MemoryError, OSError, ImportError)) or
                                 _provider(type(causal).__module__))
        causal_traceback = causal.__traceback__
        while causal_traceback is not None and causal_traceback.tb_next is not None:
            causal_traceback = causal_traceback.tb_next
        causal_module = causal_traceback.tb_frame.f_globals.get('__name__', '') if causal_traceback is not None else ''
        resource_or_provider |= _provider(causal_module)
        if causal is not error:
            causal_direct = causal_traceback is not None and Path(causal_traceback.tb_frame.f_code.co_filename).resolve() == source.resolve()
            causal_syntax = isinstance(causal, SyntaxError) and causal.filename is not None and Path(causal.filename).resolve() == source.resolve()
            if not causal_syntax and not (causal_direct and _explicit_raise(causal_traceback)):
                unknown_chain = True
        path = ancestors | {id(causal)}
        if causal.__cause__ is not None:
            pending.append((causal.__cause__, path))
        if causal.__context__ is not None and causal.__context__ is not causal.__cause__:
            pending.append((causal.__context__, path))
    if pending:
        # Cycles and an uninspected remainder cannot establish attribution.
        unknown_chain = True
    if resource_or_provider:
        return 'infrastructure', origin
    if unknown_chain:
        return 'unknown', origin
    if not isinstance(error, Exception):
        return 'unknown', origin
    if direct and not syntax:
        # A native backend may throw an exception at a CALL in submitted code
        # without adding a Python provider frame. Only an explicit raise proves
        # that a direct-source exception was issued by that source itself.
        if not _explicit_raise(traceback):
            return 'unknown', origin
    return ('candidate', 'submission') if direct or syntax else ('unknown', origin)


def _write_failure(output, source, failure):
    error = failure.error
    classification, origin = _failure_attribution(error, failure.stage, source)
    try:
        message = str(error)[:2048]
    except BaseException:
        message = '<exception message unavailable>'
    receipt = {'schema_version': 1, 'pid': os.getpid(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'stage': failure.stage, 'classification': classification,
        'error': {'type': type(error).__name__[:128], 'message': message, 'origin': origin}}
    payload = (json.dumps(receipt, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(payload) > 16384:
        raise ValueError('worker failure receipt exceeds 16 KiB')
    target = output.with_suffix('.failure.json')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{target.name}.',
                dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _EngineBoundary:
    """Apply the same direct-call boundary to warmup and cache probe calls."""
    def __init__(self, engine, vocabulary):
        self.engine, self.vocabulary = engine, vocabulary

    def _result(self, result, batch):
        # Cheap metadata checks only: no content scan, copies, or failure I/O
        # on successful timed calls. Wrong returns are known API violations.
        if torch.get_num_threads() != 2:
            raise _ContractViolation('submission changed CPU thread budget during an Engine call')
        if (not isinstance(result, torch.Tensor) or not result.is_floating_point() or
                result.ndim != 2 or tuple(result.shape) != (batch, self.vocabulary)):
            raise _ContractViolation('Engine must return floating logits with shape [batch, vocabulary]')
        return result

    def prefill(self, ids):
        batch = ids.shape[0]
        return self._result(_call('prefill', self.engine.prefill, ids), batch)

    def decode(self, token):
        batch = token.shape[0]
        return self._result(_call('decode', self.engine.decode, token), batch)


def _construct_engine(engine_class, model):
    engine = engine_class(model)
    if not all(callable(getattr(engine, method, None)) for method in ('prefill', 'decode')):
        raise _ContractViolation('Engine must define callable prefill and decode methods')
    return _EngineBoundary(engine, model.config.vocab_size)

def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
        if torch.mps.driver_allocated_memory() > 4_000_000_000:
            raise MemoryError('sampled MPS driver memory exceeds 4 GB')

def load_engine(path):
    spec = importlib.util.spec_from_file_location('submission', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = getattr(module, 'Engine', None)
    if not callable(engine):
        raise _ContractViolation('submission must define a callable Engine')
    return engine

def run(source, inputs, output, device):
    source, inputs, output = Path(source), Path(inputs), Path(output)
    try:
        return _run(source, inputs, output, device)
    except BaseException as error:
        failure = error if isinstance(error, _BoundaryFailure) else _BoundaryFailure(
            'candidate_contract' if isinstance(error, _ContractViolation) else 'worker_processing', error)
        try:
            _write_failure(output, source, failure)
        except BaseException as reporting_error:
            # A failed reporting path is intentionally an opaque infrastructure
            # exit. The parent must not turn a missing receipt into zero credit.
            print(f'worker failure reporting failed: {type(reporting_error).__name__}', file=sys.stderr)
        raise


def _run(source, inputs, output, device):
    global torch
    torch, AutoModelForCausalLM, local_model, run_cache_probes = _call('dependency_import', _dependencies)
    _call('worker_setup', torch.set_num_threads, 2)
    _call('worker_setup', torch.manual_seed, 0)
    model_path = _call('asset_lookup', local_model)
    model = _call('model_load', lambda: AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa').eval().to(device))
    init_start = time.perf_counter()
    engine_class = _call('candidate_import', load_engine, source)
    engine = _call('engine_init', _construct_engine, engine_class, model)
    engine_init_s = time.perf_counter() - init_start
    if torch.get_num_threads() != 2:
        raise _BoundaryFailure('candidate_contract', _ContractViolation('submission changed CPU thread budget'))
    task = _call('input_load', torch.load, inputs, weights_only=True)
    rows = []
    setup_start = time.perf_counter()
    with torch.inference_mode():
        # Two warm requests; startup/conversion/loading excluded from timing.
        warm_cases = []
        warm_generator = torch.Generator().manual_seed(secrets.randbits(63))
        for length in (32, 128, 512):
            # Distinct from every scored prompt, so cross-request reuse gets no
            # artificially warmed answers. Warm only shapes and dispatch paths.
            for _ in range(2):
                warm_cases.append({'ids': torch.randint(3, model.config.vocab_size,
                    (1, length), generator=warm_generator)})
        for warm in warm_cases:
            logits = engine.prefill(warm['ids'].to(device))
            engine.decode(logits.argmax(-1, keepdim=True))
        _call('device_sync', sync, device)
        warmup_s = time.perf_counter() - setup_start
        for case in task['cases']:
            ids = case['ids'].to(device)
            before = ids.clone()
            _call('device_sync', sync, device)
            start = time.perf_counter()
            logits = engine.prefill(ids)
            _call('device_sync', sync, device)
            prefill_s = time.perf_counter() - start
            first = logits.detach().float().cpu().clone()
            trace = []
            tokens = []
            _call('device_sync', sync, device)
            start = time.perf_counter()
            for _ in range(task['decode_steps']):
                token = logits.argmax(-1, keepdim=True)
                tokens.append(token)
                logits = engine.decode(token)
                trace.append(logits.clone())
            _call('device_sync', sync, device)
            decode_s = time.perf_counter() - start
            # Copy outside timer; all timed outputs checked, including every token.
            greedy = torch.stack(trace).detach().float().cpu().clone()
            token_ids = torch.stack(tokens).cpu().clone()
            engine.prefill(ids)
            forced = []
            for t in case['forced']:
                forced_token = t.to(device).clone()
                forced.append(engine.decode(forced_token).detach().float().cpu().clone())
                if not torch.equal(forced_token.cpu(), t):
                    raise _ContractViolation('forced token mutation')
            # Baseline checks its cached outputs against an uncached full-context
            # pass. The candidate receives no oracle through its Engine API;
            # filesystem confidentiality requires an external sandbox.
            if source.name == 'reference.py':
                full_ids = torch.cat([ids, torch.cat(tokens, dim=1)], dim=1)
                def reference_oracle():
                    uncached = model(input_ids=full_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu()
                    torch.testing.assert_close(uncached, greedy[-1], atol=0.002, rtol=0.002)
                    torch.testing.assert_close(model(input_ids=torch.cat([ids, case['forced'].flatten().view(1,-1).to(device)], dim=1),
                        use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu(), forced[-1], atol=0.002, rtol=0.002)
                _call('reference_oracle', reference_oracle)
            if not torch.equal(ids, before):
                raise _ContractViolation('input mutation')
            rows.append(dict(key=case['key'], length=ids.shape[1],
                prefill_s=prefill_s, decode_s=decode_s, first=first,
                greedy=greedy, tokens=token_ids, forced=torch.stack(forced)))
    memory = {'allocated_bytes': torch.mps.current_allocated_memory(),
              'driver_bytes': torch.mps.driver_allocated_memory()} if device == 'mps' else {}
    probes=[]
    if 'cache_probe_inputs' in task:
        with torch.inference_mode():
            oracle=(lambda ids:model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1,:]) if source.name=='reference.py' else None
            probes=_call('cache_probes', run_cache_probes, engine,task['cache_probe_inputs'],device,oracle)
        _call('device_sync', sync, device)
    _call('output_save', torch.save, dict(rows=rows, cache_probes=probes, warmup_s=warmup_s, engine_init_s=engine_init_s,
        mps_memory=memory, torch_threads=torch.get_num_threads()), output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    a = p.parse_args()
    run(a.source, a.inputs, a.output, a.device)
