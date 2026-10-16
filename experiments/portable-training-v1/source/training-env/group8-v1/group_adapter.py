"""Explicit group8 successor over the frozen single-candidate native adapter.

No optimizer, network client or second accelerator lease lives here. Submitted
Python retains the native trusted/cooperative boundary; a UID job sandbox is
still required before admitting arbitrary hostile recipes.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import league_adapter as native
from training_league import TrainingLeague

CONTRACT = 'training-group8-public-v1'
OBJECTIVE = 'development_cross_entropy_incumbent_group8-public-v1'
EXTRA = {'group_contract', 'group_id', 'group_slot', 'group_binding_sha256',
         'aa_path', 'aa_sha256', 'noise_gate', 'source_bindings'}


def primitive(value):
    result = {k: v for k, v in value.items() if k not in EXTRA}
    result['objective_name'] = native.MOVING_OBJECTIVE
    return result


def aa_identity(value):
    return {k: value[k] for k in ('incumbent_sha256', 'model_identity', 'oracle_sha256',
        'data_manifest_sha256', 'runtime', 'runtime_sha256', 'suite_sha256', 'budget', 'profile', 'device')}


def spread_gate(pairs):
    if len(pairs) != 10:
        raise ValueError('ten actual A/A pairs required')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for pair in pairs for x in pair):
        raise ValueError('positive finite A/A losses required')
    spread = max(abs(math.log(a) - math.log(b)) for a, b in pairs)
    return spread, max(.02, 2 * spread)


def read_aa(path, value):
    path = Path(path).resolve()
    aa = native.read_json(path)
    if (aa.get('schema') != 'training-loss-aa-v1' or aa.get('identity') != aa_identity(value)
            or aa.get('synthetic') is not False or aa.get('status') != 'ok'):
        raise ValueError('actual source/runtime/model/budget-bound A/A required')
    pairs, seeds = [], set()
    for pair in aa.get('pairs', []):
        seed = pair['seed']
        if type(seed) is not int or not 0 <= seed < 2**31 or seed in seeds:
            raise ValueError('unique A/A seeds required')
        seeds.add(seed)
        losses, initials = [], []
        for role in ('left', 'right'):
            item = pair[role]
            record_path = (path.parent / item['path']).resolve()
            if not record_path.is_relative_to(path.parent) or native.sha(record_path) != item['sha256']:
                raise ValueError('A/A proof changed or escaped')
            record = native.check_receipt(native.read_json(record_path), Path(value['incumbent_source']),
                                           primitive(value), record_path.parent, seed)
            m = record['metrics']
            if (record.get('status') != 'ok' or record.get('synthetic_metadata_control')
                    or record['provenance'].get('data_kind') != 'real'
                    or m.get('evaluation_offset_tokens', 0) != 0
                    or not m.get('initial_tensor_sha256') or not m.get('initial_state_spec_sha256')
                    or m['initial_state_spec_sha256'] != m.get('final_state_spec_sha256')
                    or not m.get('final_tensor_sha256')
                    or m['final_tensor_sha256'] != m.get('loaded_final_tensor_sha256')):
                raise ValueError('A/A independent native grade/state evidence missing')
            loss = m.get('held_out_loss')
            if type(loss) not in (int, float) or not math.isfinite(loss) or loss <= 0:
                raise ValueError('A/A loss must be finite and positive')
            if not math.isclose(record.get('reward', -1), math.exp(-loss), rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError('A/A independent reward mismatch')
            losses.append(loss)
            initials.append((m['initial_tensor_sha256'], m['initial_state_spec_sha256']))
        if initials[0] != initials[1]:
            raise ValueError('A/A initialization differs')
        pairs.append(tuple(losses))
    spread, gate = spread_gate(pairs)
    if aa.get('spread') != spread or aa.get('noise_gate') != gate:
        raise ValueError('A/A gate does not match empirical loss spread')
    if seeds & (set(value['seeds']) | set(value['validation_seeds'])):
        raise ValueError('A/A seeds cannot be training/validation seeds')
    return aa


def promotion_gate(receipt, output, gate):
    """Retain mean-CE guards; add median and >=70% per-pair sign guards."""
    import statistics
    if receipt.get('status') != 'ok' or receipt.get('comparison_eligible') is not True:
        return False
    for phase, summary in [('reward', receipt), ('validation', receipt['validation'])]:
        values = {side: {} for side in ('incumbent', 'challenger')}
        for item in receipt['native_receipts']:
            if item['phase'] == phase and item['side'] in values:
                values[item['side']][item['seed']] = native.read_json(Path(output) / item['path'])['metrics']['held_out_loss']
        a, b = values['incumbent'], values['challenger']
        if len(a) < 3 or set(a) != set(b):
            raise ValueError('at least three matched seeds per partition required')
        mean_a, mean_b = summary['incumbent_metric'], summary['challenger_metric']
        median_a, median_b = statistics.median(a.values()), statistics.median(b.values())
        if (mean_a <= 0 or median_a <= 0 or not mean_b < mean_a * (1 - gate)
                or not median_b < median_a * (1 - gate)
                or sum(b[s] < a[s] for s in a) / len(a) < .7):
            return False
    return True


def compare(round_path, candidate, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    value = native.read_json(round_path)
    # Native comparison is preserved verbatim under its own derived round hash.
    # Only this trusted translation adds group bindings and stronger promotion.
    aa = read_aa(value['aa_path'], value)
    if native.sha(value['aa_path']) != value['aa_sha256'] or aa['noise_gate'] != value['noise_gate']:
        raise ValueError('A/A identity or gate changed')
    derived = out / 'paired-native-round.json'
    native.write(derived, primitive(value))
    inner = native.compare(derived, candidate, out / 'native')
    result = dict(inner)
    result.update(objective_name=OBJECTIVE, round_sha256=native.sha(round_path),
        group_contract=CONTRACT, group_id=value['group_id'], group_slot=value['group_slot'],
        group_binding_sha256=value['group_binding_sha256'], aa_sha256=value['aa_sha256'],
        noise_gate=value['noise_gate'], source_bindings=value['source_bindings'],
        native_comparison=dict(path='native/comparison.json', sha256=native.sha(out / 'native/comparison.json')))
    result['native_receipts'] = [dict(item, path='native/' + item['path']) for item in inner['native_receipts']]
    result['promotion_ready'] = promotion_gate(inner, out / 'native', value['noise_gate'])
    result['validation'] = dict(inner['validation'], passed=result['promotion_ready'])
    native.write(out / 'comparison.json', result)
    return result


def calibrate(round_path, seeds, output):
    """Server only: twenty real native episodes, never an A/A fixture claim."""
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    value = primitive(native.read_json(round_path))
    native.validate_round(value)
    if (value['profile'] != 'standard' or len(seeds) != 10 or len(set(seeds)) != 10
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)):
        raise ValueError('standard profile and ten fresh A/A seeds required')
    if set(seeds) & (set(value['seeds']) | set(value['validation_seeds'])):
        raise ValueError('A/A seeds must be disjoint from task partitions')
    deadline = time.monotonic() + min(21600, value['comparison_timeout_seconds'])
    pairs = []
    for index, seed in enumerate(seeds):
        pair = dict(seed=seed)
        for role in (('left', 'right') if index % 2 == 0 else ('right', 'left')):
            folder = out / f'{index}-{role}'
            record = native.run_native(Path(value['incumbent_source']), value, folder, seed, deadline)
            pair[role] = dict(path=str((folder / 'record.json').relative_to(out)), sha256=native.sha(folder / 'record.json'))
            if record.get('status') != 'ok' or not native.runtime_matches(value):
                raise RuntimeError('A/A native execution or source/runtime check failed')
        pairs.append(pair)
    numbers = [(native.read_json(out / p['left']['path'])['metrics']['held_out_loss'],
                native.read_json(out / p['right']['path'])['metrics']['held_out_loss']) for p in pairs]
    spread, gate = spread_gate(numbers)
    path = out / 'aa.json'
    native.write(path, dict(schema='training-loss-aa-v1', status='ok', synthetic=False,
        identity=aa_identity(value), pairs=pairs, spread=spread, noise_gate=gate))
    read_aa(path, value)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--output', required=True)
    parser.add_argument('--aa-seeds', nargs=10, type=int)
    args = parser.parse_args()
    if args.aa_seeds is not None:
        print(calibrate(args.round, args.aa_seeds, args.output))
    else:
        if not args.candidate:
            parser.error('--candidate is required outside A/A mode')
        print(json.dumps(compare(args.round, args.candidate, args.output), allow_nan=False))
