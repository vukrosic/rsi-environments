"""Worker phase receipts with tiny dependency fixtures, no ML/GPU imports."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import worker


class TinyTensor:
    def __init__(self, shape=(1, 4)):
        self.shape = shape
        self.ndim = len(shape)

    def is_floating_point(self):
        return True

    def to(self, device):
        return self

    def argmax(self, *args, **kwargs):
        return TinyTensor((1, 1))


GOOD_ENGINE = '''class Engine:
    def __init__(self, model): self.model = model
    def prefill(self, ids): return self.model.logits
    def decode(self, token): return self.model.logits
'''


class WorkerFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / 'submission.py'
        self.source.write_text(GOOD_ENGINE)
        self.inputs = self.directory / 'inputs.pt'
        self.inputs.write_text('synthetic inputs')
        self.output = self.directory / 'output.pt'
        self.model = types.SimpleNamespace(logits=TinyTensor())
        self.model.eval = Mock(return_value=self.model)
        self.model.to = Mock(return_value=self.model)
        self.loader = types.SimpleNamespace(from_pretrained=Mock(return_value=self.model))
        self.assets = Mock(return_value=Path('/fixture/no-weights'))
        self.probes = Mock(return_value=[])
        generator = types.SimpleNamespace(manual_seed=lambda seed: None)
        self.torch = types.SimpleNamespace(
            Tensor=TinyTensor,
            float32='fixture-float32', set_num_threads=Mock(), manual_seed=Mock(),
            get_num_threads=Mock(return_value=2),
            load=Mock(return_value={'cases': [], 'decode_steps': 1}),
            Generator=lambda: generator,
            randint=lambda low, high, shape, generator: TinyTensor(shape),
            inference_mode=nullcontext,
            mps=types.SimpleNamespace(synchronize=Mock(),
                driver_allocated_memory=Mock(return_value=0),
                current_allocated_memory=Mock(return_value=0)))
        # manual_seed must return the generator for the worker's chained call.
        generator.manual_seed = lambda seed: generator
        self.model.config = types.SimpleNamespace(vocab_size=4)
        self.torch.save = Mock(side_effect=lambda value, path: Path(path).write_text(json.dumps(value)))

    def run_worker(self, device='cpu'):
        with patch.object(worker, '_dependencies', return_value=(
                self.torch, self.loader, self.assets, self.probes)), \
                patch.object(sys, 'dont_write_bytecode', True):
            return worker.run(self.source, self.inputs, self.output, device)

    def receipt(self):
        path = self.output.with_suffix('.failure.json')
        self.assertLessEqual(path.stat().st_size, 16384)
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt['schema_version'], 1)
        self.assertEqual(receipt['pid'], os.getpid())
        self.assertEqual(receipt['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertIn(receipt['stage'], worker.FAILURE_STAGES)
        self.assertLessEqual(len(receipt['error']['type']), 128)
        self.assertLessEqual(len(receipt['error']['message']), 2048)
        self.assertFalse(list(self.directory.glob('.output.failure.json.*')), 'atomic temporary files removed')
        return receipt

    def assert_failure(self, stage, classification, origin=None, device='cpu'):
        with self.assertRaises((worker._BoundaryFailure, worker._ContractViolation)):
            self.run_worker(device)
        receipt = self.receipt()
        self.assertEqual(receipt['stage'], stage)
        self.assertEqual(receipt['classification'], classification)
        if origin is not None:
            self.assertEqual(receipt['error']['origin'], origin)
        return receipt

    def test_success_preserves_output_and_has_no_failure_receipt(self):
        with patch.object(worker, '_write_failure') as failure_io:
            self.run_worker()
        failure_io.assert_not_called()
        self.assertTrue(self.output.is_file())
        self.assertFalse(self.output.with_suffix('.failure.json').exists())
        self.assertEqual(json.loads(self.output.read_text())['rows'], [])

    def test_dependency_import_failure_is_reported_before_engine(self):
        with patch.object(worker, '_dependencies', side_effect=ImportError('fixture missing dependency')):
            with self.assertRaises(worker._BoundaryFailure):
                worker.run(self.source, self.inputs, self.output, 'cpu')
        receipt = self.receipt()
        self.assertEqual(receipt['stage'], 'dependency_import')
        self.assertEqual(receipt['classification'], 'infrastructure')
        self.assets.assert_not_called()

    def test_asset_lookup_failure_is_infrastructure_before_import(self):
        self.source.write_text("raise AssertionError('submitted module must not be imported')\n")
        self.assets.side_effect = RuntimeError('fixture asset failure')
        self.assert_failure('asset_lookup', 'infrastructure', 'worker')
        self.loader.from_pretrained.assert_not_called()

    def test_model_load_failure_is_infrastructure(self):
        self.loader.from_pretrained.side_effect = RuntimeError('fixture model provider failure')
        self.assert_failure('model_load', 'infrastructure')

    def test_worker_setup_failure_is_infrastructure(self):
        self.torch.manual_seed.side_effect = RuntimeError('fixture setup failure')
        self.assert_failure('worker_setup', 'infrastructure')

    def test_input_deserialization_failure_is_infrastructure(self):
        self.torch.load.side_effect = OSError('fixture corrupt input')
        self.assert_failure('input_load', 'infrastructure')

    def test_output_serialization_failure_is_infrastructure(self):
        self.torch.save.side_effect = PermissionError('fixture output path failure')
        self.assert_failure('output_save', 'infrastructure')

    def test_device_sync_failure_is_infrastructure_with_mocked_device(self):
        self.torch.mps.synchronize.side_effect = RuntimeError('fixture backend synchronization failure')
        self.assert_failure('device_sync', 'infrastructure', device='mps')

    def test_direct_submitted_import_and_constructor_exceptions_are_candidate(self):
        for stage, source in (
            ('candidate_import', "raise RuntimeError('fixture module exception')\n"),
            ('engine_init', "class Engine:\n    def __init__(self, model): raise RuntimeError('fixture Engine exception')\n")):
            with self.subTest(stage=stage):
                self.source.write_text(source)
                self.assert_failure(stage, 'candidate', 'submission')

    def test_syntax_error_is_candidate_import_failure(self):
        self.source.write_text('class Engine(:\n')
        self.assert_failure('candidate_import', 'candidate', 'submission')

    def test_missing_engine_api_is_explicit_contract_failure(self):
        for source in ('x = 1\n', 'class Engine:\n    def __init__(self, model): pass\n'):
            with self.subTest(source=source):
                self.source.write_text(source)
                self.assert_failure('candidate_contract', 'candidate', 'contract')

    def test_wrong_engine_return_is_explicit_contract_failure(self):
        for logits in (None, 1, TinyTensor((1, 3)), TinyTensor((1, 1, 4))):
            with self.subTest(logits=logits):
                self.model.logits = logits
                self.assert_failure('candidate_contract', 'candidate', 'contract')

    def test_cache_probe_contract_exception_keeps_contract_attribution(self):
        class FixtureProbeContract(ValueError):
            pass
        self.torch.load.return_value['cache_probe_inputs'] = {}
        self.probes.side_effect = FixtureProbeContract('fixture prompt mutation')
        with patch.object(worker, '_CacheContractViolation', FixtureProbeContract):
            self.assert_failure('candidate_contract', 'candidate', 'contract')

    def test_unattributed_cache_probe_exception_stays_infrastructure(self):
        self.torch.load.return_value['cache_probe_inputs'] = {}
        self.probes.side_effect = RuntimeError('fixture external probe failure')
        self.assert_failure('cache_probes', 'infrastructure')

    def test_direct_prefill_and_decode_exceptions_keep_exact_api_phase(self):
        for method in ('prefill', 'decode'):
            with self.subTest(method=method):
                self.source.write_text(GOOD_ENGINE.replace(
                    f'def {method}(self, ' + ('ids' if method == 'prefill' else 'token') + '): return self.model.logits',
                    f'def {method}(self, ' + ('ids' if method == 'prefill' else 'token') + "): raise ValueError('fixture API exception')"))
                self.assert_failure(method, 'candidate', 'submission')

    def test_resource_failure_inside_engine_is_infrastructure(self):
        self.source.write_text("class Engine:\n    def __init__(self, model): raise MemoryError('fixture OOM')\n")
        self.assert_failure('engine_init', 'infrastructure', 'submission')

    def test_lazy_thread_change_is_explicit_contract_violation(self):
        for method, argument in (('prefill', 'ids'), ('decode', 'token')):
            with self.subTest(method=method):
                self.torch.get_num_threads.return_value = 2
                self.model.change_threads = Mock(side_effect=lambda:
                    setattr(self.torch.get_num_threads, 'return_value', 3))
                self.source.write_text(GOOD_ENGINE.replace(
                    f'def {method}(self, {argument}): return self.model.logits',
                    f'def {method}(self, {argument}): self.model.change_threads(); return self.model.logits'))
                self.assert_failure('candidate_contract', 'candidate', 'contract')

    def test_external_provider_runtime_error_inside_engine_is_infrastructure(self):
        provider = {'__name__': 'torch.fixture_backend'}
        exec(compile("def fail(): raise RuntimeError('fixture provider runtime error')\n",
                     str(self.directory / 'provider.py'), 'exec'), provider)
        self.model.fail = provider['fail']
        self.source.write_text("class Engine:\n    def __init__(self, model): model.fail()\n")
        self.assert_failure('engine_init', 'infrastructure', 'provider')

    def test_native_exceptions_at_submitted_call_are_unknown(self):
        for error_type in ('RuntimeError', 'ValueError', 'TypeError'):
            with self.subTest(error_type=error_type):
                self.source.write_text('''import ctypes
class Engine:
    def __init__(self, model):
        ctypes.pythonapi.PyErr_SetString.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ctypes.pythonapi.PyErr_SetString(ERROR_TYPE, b'fixture native backend error')
'''.replace('ERROR_TYPE', error_type))
                self.assert_failure('engine_init', 'unknown', 'submission')

    def test_wrapped_provider_oom_is_not_a_candidate_exception(self):
        self.source.write_text('''class Engine:
    def __init__(self, model):
        try: raise MemoryError('fixture provider OOM')
        except MemoryError as error: raise RuntimeError('wrapped failure') from error
''')
        self.assert_failure('engine_init', 'infrastructure')

    def test_wrapped_provider_runtime_error_is_not_a_candidate_exception(self):
        provider = {'__name__': 'torch.fixture_backend'}
        exec(compile("def fail(): raise RuntimeError('fixture provider error')\n",
                     str(self.directory / 'provider.py'), 'exec'), provider)
        self.model.fail = provider['fail']
        self.source.write_text('''class Engine:
    def __init__(self, model):
        try: model.fail()
        except RuntimeError as error: raise RuntimeError('wrapped failure') from error
''')
        self.assert_failure('engine_init', 'infrastructure')

    def test_explicit_cause_does_not_hide_provider_context(self):
        provider = {'__name__': 'torch.fixture_backend'}
        exec(compile("def fail(): raise RuntimeError('fixture provider error')\n",
                     str(self.directory / 'provider.py'), 'exec'), provider)
        self.model.fail = provider['fail']
        self.source.write_text('''class Engine:
    def __init__(self, model):
        try: raise ValueError('independent direct cause')
        except ValueError as error: saved = error
        try: model.fail()
        except RuntimeError: raise RuntimeError('wrapped failure') from saved
''')
        self.assert_failure('engine_init', 'infrastructure')

    def test_wrapped_native_call_exception_remains_unknown(self):
        self.source.write_text('''import ctypes
class Engine:
    def __init__(self, model):
        ctypes.pythonapi.PyErr_SetString.argtypes = [ctypes.py_object, ctypes.c_char_p]
        try: ctypes.pythonapi.PyErr_SetString(RuntimeError, b'fixture native error')
        except RuntimeError as error: raise ValueError('wrapped native failure') from error
''')
        self.assert_failure('engine_init', 'unknown')

    def test_direct_exception_chains_are_bounded_and_cycles_unknown(self):
        for depth, classification in ((1, 'candidate'), (17, 'unknown')):
            with self.subTest(depth=depth):
                self.source.write_text('''def fail(depth):
    if depth == 0: raise RuntimeError('direct failure')
    try: fail(depth-1)
    except RuntimeError as error: raise RuntimeError('direct wrapper') from error
class Engine:
    def __init__(self, model): fail(DEPTH)
'''.replace('DEPTH', str(depth)))
                self.assert_failure('engine_init', classification)
        self.source.write_text('''class Engine:
    def __init__(self, model):
        error = RuntimeError('cyclic failure')
        error.__cause__ = error
        raise error
''')
        self.assert_failure('engine_init', 'unknown')

    def test_reference_exception_is_infrastructure(self):
        self.source = self.directory / 'reference.py'
        self.source.write_text("class Engine:\n    def __init__(self, model): raise RuntimeError('fixture reference failure')\n")
        self.assert_failure('engine_init', 'infrastructure', 'submission')

    def test_cache_probe_api_call_uses_same_attribution(self):
        self.source.write_text('''class Engine:
    def __init__(self, model): self.model, self.calls = model, 0
    def prefill(self, ids):
        self.calls += 1
        if self.calls > 6: raise RuntimeError('fixture cache-probe Engine failure')
        return self.model.logits
    def decode(self, token): return self.model.logits
''')
        self.torch.load.return_value['cache_probe_inputs'] = {}
        self.probes.side_effect = lambda engine, *args: engine.prefill(TinyTensor())
        self.assert_failure('prefill', 'candidate', 'submission')

    def test_receipt_diagnostic_unicode_size_is_bounded(self):
        self.source.write_text("class Engine:\n    def __init__(self, model): raise RuntimeError('🔥' * 10000)\n")
        receipt = self.assert_failure('engine_init', 'candidate')
        self.assertEqual(len(receipt['error']['message']), 2048)


if __name__ == '__main__':
    unittest.main()
