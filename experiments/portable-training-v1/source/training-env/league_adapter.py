"""Finite TinyStories comparisons using independent native evaluator subprocesses.

Trusted-local fault containment, not a sandbox. Standard native calibration and
per-run accelerator leases remain intact. Lease gaps do not establish continuous
idle timing. Budgets fix optimizer calls/tokens/resource limits, not equal FLOPs.
Validation uses predeclared disjoint dev windows and training seeds.
Final is never graded here.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import platform
from importlib import metadata
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
NATIVE = ('evaluate.py', 'worker.py', 'grader.py', 'protocol.py', 'checks.py', 'data.py', 'tensor_hashes.py', 'calibrate.py', 'process_control.py', 'starter.py')
MOVING_POLICY = 'training-incumbent-loss-v2'
MOVING_OBJECTIVE = 'development_cross_entropy_incumbent_v2'


def moving_round(r):
    return r.get('reward_policy') == MOVING_POLICY


def relative_improvement(incumbent, challenger):
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (incumbent, challenger)):
        raise ValueError('finite nonnegative paired losses required')
    if incumbent == 0:
        return 0.0 if challenger == 0 else None
    value = 1.0 - challenger / incumbent
    return value if math.isfinite(value) else None


def verified_two_percent_win(incumbent, challenger):
    relative_improvement(incumbent, challenger)
    return incumbent > 0 and challenger < incumbent and challenger <= incumbent * 0.98


def pair_order(index, moving=False):
    return ('challenger', 'incumbent') if moving and index % 2 else ('incumbent', 'challenger')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def runtime_identity(device):
    """Current interpreter/runtime identity without importing accelerator APIs."""
    return {'python': sys.version.split()[0], 'executable': sys.executable,
            'torch': metadata.version('torch'), 'numpy': metadata.version('numpy'),
            'hardware': platform.platform(), 'machine': platform.machine(),
            'host_sha256': hashlib.sha256(platform.node().encode()).hexdigest(),
            'device': device}


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError('JSON input exceeds 1 MB')
    value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(value, dict):
        raise ValueError('JSON input must be an object')
    return value


def runtime_matches(r):
    return (r.get('runtime_sha256') == {n: sha(HERE / n) for n in NATIVE}
            and r.get('suite_sha256') == {n: sha(HERE.parent / 'suite' / n) for n in ('records.py', 'resources.py')}
            and r.get('runtime') == runtime_identity(r['device']))


def check_receipt(value, source, r, out, seed):
    metrics, provenance = value['metrics'], value['provenance']
    if any(value.get(k) != v for k, v in {'schema_version': 1, 'environment': 'training', 'task': 'tinystories-next-token', 'profile': r['profile']}.items()):
        raise RuntimeError('native envelope identity mismatch')
    if metrics.get('protocol') != r['budget'] or provenance.get('device') != r['device'] or provenance.get('seed') != seed or metrics.get('split') != 'dev':
        raise RuntimeError('native protocol/device/seed/dev binding mismatch')
    actual_runtime = runtime_identity(r['device'])
    if r.get('runtime') != actual_runtime or any(provenance.get(k) != actual_runtime[k] for k in ('python', 'torch', 'numpy', 'hardware', 'machine', 'device')):
        raise RuntimeError('native runtime descriptor mismatch')
    expected_native = {n: r['runtime_sha256'][n] for n in NATIVE[:7]}
    if provenance.get('evaluator_sha256') != expected_native or provenance.get('suite_sha256') != r['suite_sha256']:
        raise RuntimeError('native receipt runtime identity mismatch')
    if provenance.get('source_sha256') != sha(source) or sha(out / 'candidate.py') != sha(source):
        raise RuntimeError('native candidate snapshot identity mismatch')
    if provenance.get('data_manifest_sha256') != r['data_manifest_sha256']:
        raise RuntimeError('native data manifest identity mismatch')
    if value['status'] == 'ok':
        budget = r['budget']
        expected = {'steps': budget['steps'], 'training_tokens': budget['steps'] * budget['batch'] * budget['context'], 'eval_tokens': budget['eval_batches'] * budget['batch'] * budget['context']}
        if any(type(metrics.get(k)) is not int or metrics[k] != v for k, v in expected.items()):
            raise RuntimeError('native actual call/token budget mismatch')
        if type(metrics.get('params')) is not int or not 0 < metrics['params'] <= budget['max_params']:
            raise RuntimeError('native parameter budget mismatch')
        if provenance.get('checkpoint_sha256') != sha(out / 'weights.pt') or provenance.get('initial_checkpoint_sha256') != sha(out / 'initial-weights.pt'):
            raise RuntimeError('native model weight file evidence mismatch')
    return value


def model_identity(source):
    """Freeze model definitions/imports; only optimization code/constants vary."""
    tree = ast.parse(Path(source).read_text())
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {'build_optimizer', 'train_step'}:
            continue
        if isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if names and names <= {'LR', 'WARMUP_FRAC', 'WD'}:
                continue
        kept.append(node)
    return hashlib.sha256(ast.dump(ast.Module(body=kept, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def validate_round(r):
    allowed = {'schema_version', 'environment', 'round_id', 'workload_id', 'incumbent_source', 'incumbent_sha256', 'data_dir', 'data_manifest_sha256', 'profile', 'device', 'seeds', 'validation_seeds', 'validation_offset_tokens', 'budget', 'runtime_sha256', 'suite_sha256', 'min_improvement', 'comparison_timeout_seconds', 'calibration', 'model_identity', 'oracle_sha256', 'objective_name', 'lower_is_better', 'runtime', 'reward_policy', 'min_relative_improvement', 'anchor_source', 'anchor_sha256'}
    if set(r) - allowed:
        raise ValueError('unknown unused round fields: ' + ', '.join(sorted(set(r) - allowed)))
    if r.get('lower_is_better', True) is not True:
        raise ValueError('lower_is_better must be literal true')
    if r.get('reward_policy') not in (None, 'league-relative-v1', MOVING_POLICY):
        raise ValueError('unsupported reward policy; use an explicit new lineage')
    if r.get('objective_name', 'development_cross_entropy') != (MOVING_OBJECTIVE if moving_round(r) else 'development_cross_entropy'):
        raise ValueError('unsupported objective_name')
    if moving_round(r):
        if type(r.get('min_relative_improvement')) not in (int, float) or r['min_relative_improvement'] != 0.02:
            raise ValueError('moving incumbent promotion threshold must be 2 percent')
        if (not isinstance(r.get('anchor_source'), str) or not Path(r['anchor_source']).is_absolute()
                or r.get('anchor_sha256') != sha(HERE / 'starter.py')
                or sha(r['anchor_source']) != r['anchor_sha256']):
            raise ValueError('frozen native starter anchor required')
        if r.get('min_improvement') != 0 or type(r.get('min_improvement')) is bool:
            raise ValueError('new lineage uses relative threshold, not legacy absolute improvement')
    elif any(k in r for k in ('min_relative_improvement', 'anchor_source', 'anchor_sha256')):
        raise ValueError('new reward fields require the new reward lineage')
    if r.get('schema_version') != 1 or r.get('environment') != 'training':
        raise ValueError('unsupported round schema/environment')
    for name in ('round_id', 'workload_id', 'incumbent_source', 'incumbent_sha256', 'data_dir', 'data_manifest_sha256', 'model_identity', 'oracle_sha256'):
        if not isinstance(r.get(name), str) or not r[name]:
            raise ValueError('missing round field: ' + name)
    if r['workload_id'] != 'tinystories-next-token' or r.get('profile') not in {'smoke', 'standard'}:
        raise ValueError('unsupported workload/profile')
    if r.get('device') not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('unsupported device')
    for name in ('seeds', 'validation_seeds'):
        values = r.get(name)
        if not isinstance(values, list) or not 1 <= len(values) <= 8 or any(type(v) is not int or not 0 <= v < 2**31 for v in values) or len(set(values)) != len(values):
            raise ValueError('invalid predeclared seed partition: ' + name)
    if set(r['seeds']) & set(r['validation_seeds']):
        raise ValueError('reward and validation seeds overlap')
    if r.get('runtime') != runtime_identity(r['device']):
        raise ValueError('runtime descriptor identity mismatch')
    for name in ('min_improvement', 'comparison_timeout_seconds'):
        v = r.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or (v < 0 if name == 'min_improvement' and moving_round(r) else v <= 0):
            raise ValueError('invalid ' + name)
    if r['comparison_timeout_seconds'] > 43200:
        raise ValueError('comparison timeout exceeds finite 12 hour cap')
    from protocol import get_protocol
    protocol = get_protocol(r['profile'])
    offset = r.get('validation_offset_tokens')
    if type(offset) is not int or offset < protocol.eval_batches * protocol.batch * (protocol.context + 1) or offset % (protocol.context + 1):
        raise ValueError('validation dev offset must be aligned and disjoint from reward window')
    if r.get('budget') != protocol.record():
        raise ValueError('budget must exactly match native protocol')
    if r.get('runtime_sha256') != {n: sha(HERE / n) for n in NATIVE}:
        raise ValueError('native runtime identity mismatch')
    if r.get('suite_sha256') != {n: sha(HERE.parent / 'suite' / n) for n in ('records.py', 'resources.py')}:
        raise ValueError('suite runtime identity mismatch')
    return r


def run_native(source, r, out, seed, deadline, offset=0):
    """Own and bound the entire evaluator tree, preserving native resource guards."""
    args = [sys.executable, str(HERE / 'evaluate.py'), '--candidate', str(source), '--data', r['data_dir'], '--output', str(out), '--profile', r['profile'], '--device', r['device'], '--seed', str(seed), '--split', 'dev', '--state-evidence', '--evaluation-offset-tokens', str(offset)]
    if r.get('calibration'):
        args += ['--calibration', r['calibration']]
    from process_control import run_command, CommandTimeout
    try:
        execution = run_command(args, cwd=out.parent, env=os.environ.copy(), timeout=max(0.001, deadline - time.monotonic()))
    except CommandTimeout as exc:
        write(out.with_suffix('.supervisor.json'), exc.supervisor_metadata)
        raise RuntimeError('comparison wall-time budget exhausted') from exc
    out.with_suffix('.stdout.log').write_text(execution.stdout)
    out.with_suffix('.stderr.log').write_text(execution.stderr)
    write(out.with_suffix('.supervisor.json'), execution.supervisor_metadata)
    if execution.supervisor_metadata.get('cleanup_errors') or execution.supervisor_metadata.get('discovery_errors'):
        raise ValueError('native supervisor discovery/cleanup errors')
    record = out / 'record.json'
    if not record.exists() or record.stat().st_size > 1_000_000:
        raise RuntimeError('native evaluator omitted bounded receipt')
    value = read_json(record)
    if execution.returncode != 0 and value.get('status') == 'ok':
        raise ValueError('nonzero native exit with claimed ok receipt')
    return check_receipt(value, source, r, out, seed)


def compare(round_path, candidate, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        r = read_json(round_path)
    except Exception as exc:
        result = {'schema_version': 1, 'environment': 'training', 'status': 'error', 'relative_reward': None, 'comparison_eligible': False, 'exclusion_reasons': [str(exc)]}
        write(out / 'comparison.json', result)
        return result
    result = dict(schema_version=1, environment='training', round_id=r.get('round_id'), workload_id=r.get('workload_id'), objective_name='development_cross_entropy', lower_is_better=True, profile=r.get('profile'), device=r.get('device'), status='error', incumbent_metric=None, challenger_metric=None, correctness_pass=False, comparison_eligible=False, exclusion_reasons=[], relative_reward=None, relative_reward_version='league-relative-v1', native_receipts=[], validation={'passed': False, 'corpus': 'disjoint predeclared native dev windows and training seeds'}, round_sha256=sha(round_path))
    if moving_round(r):
        result.update(objective_name=MOVING_OBJECTIVE, relative_reward_version=MOVING_POLICY,
                      reward_policy=MOVING_POLICY, min_relative_improvement=r.get('min_relative_improvement'),
                      anchor_sha256=r.get('anchor_sha256'), pair_orders=[], promotion_ready=False)
    # Echo the frozen inputs even on invalid/blocked terminal paths so the
    # shared controller can authenticate the excluded envelope.
    for field in ('budget', 'runtime_sha256', 'suite_sha256', 'data_manifest_sha256', 'seeds', 'validation_seeds', 'oracle_sha256', 'model_identity', 'validation_offset_tokens', 'runtime'):
        if field in r:
            result[field] = r[field]
    runtime_initial_matched = False
    try:
        validate_round(r)
        runtime_initial_matched = runtime_matches(r)
        write(out / 'round.json', r)
        sources = {}
        origins = [('incumbent', r['incumbent_source']), ('challenger', candidate)]
        if moving_round(r):
            origins.append(('anchor', r['anchor_source']))
        for side, origin in origins:
            path = out / (side + '.py')
            path.write_bytes(Path(origin).read_bytes())
            sources[side] = path
            result[side + '_source_sha256'] = sha(path)
        if result['incumbent_source_sha256'] != r['incumbent_sha256']:
            raise RuntimeError('incumbent frozen identity mismatch')
        if moving_round(r):
            result.update(objective_name=MOVING_OBJECTIVE, relative_reward_version=MOVING_POLICY,
                          reward_policy=MOVING_POLICY, min_relative_improvement=0.02,
                          anchor_sha256=r['anchor_sha256'], pair_orders=[], promotion_ready=False)
            if result['anchor_source_sha256'] != r['anchor_sha256']:
                raise RuntimeError('anchor snapshot identity mismatch')
        try:
            fixed_model = model_identity(sources['incumbent'])
        except SyntaxError as exc:
            raise RuntimeError('frozen incumbent syntax invalid') from exc
        if r.get('model_identity') != fixed_model or r.get('oracle_sha256') != r['runtime_sha256'].get('grader.py'):
            raise RuntimeError('declared model/oracle identity mismatch')
        data = Path(r['data_dir'])
        if sha(data / 'manifest.json') != r['data_manifest_sha256']:
            raise RuntimeError('data manifest identity mismatch')
        if fixed_model != model_identity(sources['challenger']):
            result.update(status='invalid', relative_reward=0.0)
            result['exclusion_reasons'].append('candidate changed frozen model definition')
            return result
        if moving_round(r):
            if model_identity(sources['anchor']) != fixed_model:
                raise RuntimeError('anchor and incumbent must preserve the same native model')
            if result['challenger_source_sha256'] == result['incumbent_source_sha256']:
                result.update(status='invalid', relative_reward=0.0)
                result['exclusion_reasons'].append('candidate no-op: source equals current incumbent')
                return result
        result.update(model_identity=model_identity(sources['incumbent']), oracle_sha256=r['runtime_sha256'].get('grader.py'), validation_offset_tokens=r['validation_offset_tokens'], comparison_timeout_seconds=r['comparison_timeout_seconds'], min_improvement=r['min_improvement'], data_manifest_sha256=r['data_manifest_sha256'], runtime_sha256=r['runtime_sha256'], suite_sha256=r['suite_sha256'], budget=r['budget'], seeds=r['seeds'], validation_seeds=r['validation_seeds'], measurement_order='fixed incumbent then challenger; quality comparison, no timing comparison')
        deadline = time.monotonic() + r['comparison_timeout_seconds']
        losses = {phase: {side: [] for side in sources} for phase in ('reward', 'validation')}
        init_proven = True
        pair_index = 0
        for phase, seeds in [('reward', r['seeds']), ('validation', r['validation_seeds'])]:
            for seed in seeds:
                pair = {}
                order = list(pair_order(pair_index, moving_round(r)))
                if moving_round(r):
                    result['pair_orders'].append(dict(phase=phase, seed=seed, order=list(order)))
                    order.append('anchor')
                for side in order:
                    source = sources[side]
                    native_out = out / f'{phase}-{seed}-{side}'
                    record = run_native(source, r, native_out, seed, deadline, r['validation_offset_tokens'] if phase == 'validation' else 0)
                    result['native_receipts'].append({'path': str(native_out.relative_to(out) / 'record.json'), 'sha256': sha(native_out / 'record.json'), 'side': side, 'phase': phase, 'seed': seed, 'native_reward': record.get('reward'), 'checkpoint_sha256': record['provenance'].get('checkpoint_sha256')})
                    if not runtime_matches(r):
                        raise RuntimeError('trusted runtime drift during execution')
                    if record['status'] != 'ok':
                        result['status'] = record['status'] if side == 'challenger' else 'blocked'
                        if moving_round(r) and result['status'] == 'invalid':
                            reason = str(record['metrics'].get('reason', '')).lower()
                            if any(x in reason for x in ('memory', 'oom', 'cuda', 'driver', 'backend', 'device',
                                                          'data changed', 'source changed', 'manifest', 'timeout', 'cleanup')):
                                result['status'] = 'blocked'
                        result['relative_reward'] = 0.0 if result['status'] == 'invalid' else None
                        result['exclusion_reasons'].append(f'{side} native {record["status"]}: {record["metrics"].get("reason")}')
                        return result
                    metrics = record['metrics']
                    expected_offset = r['validation_offset_tokens'] if phase == 'validation' else 0
                    if metrics.get('evaluation_offset_tokens', 0) != expected_offset:
                        raise RuntimeError('native dev window binding mismatch')
                    loss = metrics.get('held_out_loss')
                    if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
                        raise RuntimeError('native grader loss invalid')
                    if record['provenance'].get('data_kind') != 'real':
                        result['exclusion_reasons'].append('non-real data excluded')
                    if not metrics.get('final_tensor_sha256') or metrics.get('final_tensor_sha256') != metrics.get('loaded_final_tensor_sha256'):
                        raise RuntimeError('trained/grader model tensor binding missing or different')
                    losses[phase][side].append(loss)
                    pair[side] = (metrics.get('initial_tensor_sha256'), metrics.get('initial_state_spec_sha256'))
                    if not all(pair[side]) or not metrics.get('final_state_spec_sha256'):
                        raise RuntimeError('native initial model/state specification proof missing')
                    if metrics['initial_state_spec_sha256'] != metrics['final_state_spec_sha256']:
                        result.update(status='invalid' if side == 'challenger' else 'blocked', relative_reward=0.0 if side == 'challenger' else None)
                        result['exclusion_reasons'].append('fixed model state specification changed')
                        return result
                    if sha(source) != result[side + '_source_sha256']:
                        raise RuntimeError('source changed during execution')
                if pair['incumbent'] != pair['challenger']:
                    result.update(status='invalid', relative_reward=0.0)
                    result['exclusion_reasons'].append('challenger seeded initial model differs from incumbent')
                    return result
                if moving_round(r) and pair['anchor'] != pair['incumbent']:
                    raise RuntimeError('anchor seeded initial state differs from the matched pair')
                pair_index += 1
        if sha(data / 'manifest.json') != r['data_manifest_sha256']:
            raise RuntimeError('data manifest changed during comparison')
        if r['runtime_sha256'] != {n: sha(HERE / n) for n in NATIVE}:
            raise RuntimeError('native runtime changed during comparison')
        if r['suite_sha256'] != {n: sha(HERE.parent / 'suite' / n) for n in ('records.py', 'resources.py')}:
            raise RuntimeError('suite runtime changed during comparison')
        a, b = (math.fsum(v / len(r['seeds']) for v in losses['reward'][side]) for side in ('incumbent', 'challenger'))
        result.update(status='ok', incumbent_metric=a, challenger_metric=b, correctness_pass=True)
        if r['profile'] != 'standard':
            result['exclusion_reasons'].append('smoke diagnostic excluded')
        if not init_proven:
            result['exclusion_reasons'].append('native seeded initial model weight proof missing or different')
        eligible = not result['exclusion_reasons']
        result['comparison_eligible'] = eligible
        result['relative_reward'] = (0.5 if max(a, b) == 0 else (a / max(a, b)) / (a / max(a, b) + b / max(a, b))) if eligible else None
        va, vb = (math.fsum(v / len(r['validation_seeds']) for v in losses['validation'][side]) for side in ('incumbent', 'challenger'))
        result['validation'].update(incumbent_metric=va, challenger_metric=vb, passed=eligible and va - vb > r['min_improvement'], seeds=r['validation_seeds'], challenger_source_sha256=result['challenger_source_sha256'], partition={'split': 'dev', 'offset_tokens': r['validation_offset_tokens'], 'raw_span_tokens': r['budget']['eval_batches'] * r['budget']['batch'] * (r['budget']['context'] + 1)}, meaningful_improvement=va - vb, success=eligible and va - vb > r['min_improvement'])
        result['meaningful_improvement'] = a - b
        result['promotion_ready'] = eligible and a - b > r['min_improvement'] and result['validation']['passed']
        if moving_round(r):
            reward_win = verified_two_percent_win(a, b)
            validation_win = verified_two_percent_win(va, vb)
            result.update(measurement_order='alternating incumbent/challenger per matched seed; anchor reported only',
                          relative_improvement=relative_improvement(a, b),
                          promotion_ready=eligible and reward_win and validation_win)
            result['validation'].update(passed=eligible and validation_win, success=eligible and validation_win,
                                        relative_improvement=relative_improvement(va, vb), min_relative_improvement=0.02)
            result['anchor'] = dict(source_sha256=r['anchor_sha256'], report_only=True,
                reward_mean_loss=math.fsum(v / len(r['seeds']) for v in losses['reward']['anchor']),
                validation_mean_loss=math.fsum(v / len(r['validation_seeds']) for v in losses['validation']['anchor']),
                native_rewards_are_training_reward=False)
    except SyntaxError as exc:
        result.update(status='invalid', relative_reward=0.0)
        result['exclusion_reasons'].append(f'candidate syntax invalid: {exc}')
    except (RuntimeError, FileNotFoundError) as exc:
        result.update(status='blocked', relative_reward=None)
        result['exclusion_reasons'].append(str(exc))
    except Exception as exc:
        result.update(status='error', relative_reward=None)
        result['exclusion_reasons'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if runtime_initial_matched and not runtime_matches(r):
            result.update(status='blocked', relative_reward=None, comparison_eligible=False, promotion_ready=False)
            result['validation']['passed'] = False
            result['exclusion_reasons'].append('trusted runtime drift invalidates entire comparison')
        write(out / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.round, args.candidate, args.output)
    print(json.dumps(result, allow_nan=False))
    return 0  # A persisted envelope is transport success; status governs eligibility.


if __name__ == '__main__':
    raise SystemExit(main())
