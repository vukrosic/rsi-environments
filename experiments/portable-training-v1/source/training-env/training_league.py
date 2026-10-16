"""Moving-loss Training lane over the existing shared league store.

This facade adds Training evidence/promotion gates and derived reporting only.
Artifacts, sealed rounds/attempts, execution and champion CAS stay in suite.league.
It does not make arbitrary submitted Python safe; sources remain cooperative.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from suite.league.controller import League, read, sha, write, canonical
from suite.league.episode import Episode
from league_adapter import (MOVING_POLICY, MOVING_OBJECTIVE, moving_round,
                            validate_round, verified_two_percent_win, read_json,
                            NATIVE, runtime_identity, model_identity)


class TrainingLeague(League):
    def moving_metadata(self, rid):
        """Inspect sealed routing fields before applying Training-only checks.

        Shared state may also contain historical Training or other environment
        lanes. Those retain their own validators and are not migrated here.
        """
        if not isinstance(rid, str) or len(rid) != 32 or any(c not in '0123456789abcdef' for c in rid):
            raise ValueError('invalid round id')
        directory = self.root / 'rounds' / rid
        if sha(directory / 'round.json') != read(directory / 'seal.json')['round_sha256']:
            raise ValueError('round mutated')
        value = read(directory / 'round.json')
        if sha(directory / 'native.json') != value['native_file_sha256']:
            raise ValueError('native round mutated')
        native = value['native_round']
        if native.get('environment') != 'training' or not moving_round(native):
            return None
        return self.round(rid)

    def reject_reused_seeds(self, native, exclude_round=None):
        requested = set(native['seeds']) | set(native['validation_seeds'])
        lane = self.lane_identity(native)
        for directory in (self.root / 'rounds').glob('*'):
            if directory.name == exclude_round:
                continue
            saved = self.moving_metadata(directory.name)
            if saved is not None and saved['lane_id'] == lane:
                prior = saved['native_round']
                if requested & (set(prior['seeds']) | set(prior['validation_seeds'])):
                    raise ValueError('fresh reward/validation seeds required; prior round already reserved this seed')

    def freeze(self, native_round, adapter, *, max_actions=1, timeout_seconds=4800, synthetic=False):
        if max_actions != 1 or type(max_actions) is not int:
            raise ValueError('moving Training rounds reserve one candidate and fresh validation seeds')
        self.reject_reused_seeds(native_round)
        return super().freeze(native_round, adapter, max_actions=max_actions,
                              timeout_seconds=timeout_seconds, synthetic=synthetic)

    def execute(self, rid, candidate):
        with self.locked():
            r = self.round(rid)
            self.reject_reused_seeds(r['native_round'], exclude_round=rid)
            current = read(self.root / 'champions' / (r['lane_id'] + '.json'))
            if current != r['champion']:
                raise ValueError('stale incumbent; freeze a new round before native execution')
        return super().execute(rid, candidate)

    @staticmethod
    def validate_native(native, files, adapter):
        League.validate_native(native, files, adapter)
        if not moving_round(native) or native.get('objective_name') != MOVING_OBJECTIVE:
            raise ValueError('explicit new Training reward lineage required')
        validate_round(native)
        required = [HERE / 'league_adapter.py', Path(__file__),
                    HERE.parent / 'suite/__init__.py', HERE.parent / 'suite/league/__init__.py',
                    HERE.parent / 'suite/league/controller.py', HERE.parent / 'suite/league/episode.py']
        if any(files.get(str(p.resolve())) != sha(p) for p in required):
            raise ValueError('Training facade, adapter and shared store sources must be pinned')
        if files.get(str(Path(native['anchor_source']).resolve())) != native['anchor_sha256']:
            raise ValueError('frozen report-only anchor must be pinned')

    @staticmethod
    def verify_pair_evidence(r, receipt, output):
        """Recompute loss means/promotion from actual source-bound native records."""
        native = r['native_round']
        if (receipt.get('reward_policy') != MOVING_POLICY
                or receipt.get('relative_reward_version') != MOVING_POLICY
                or receipt.get('min_relative_improvement') != 0.02
                or receipt.get('anchor_sha256') != native['anchor_sha256']):
            raise ValueError('reward lineage/anchor/threshold differs')
        if receipt.get('status') != 'ok':
            return
        losses = {phase: {side: [] for side in ('incumbent', 'challenger', 'anchor')}
                  for phase in ('reward', 'validation')}
        initials, seen, actual_order = {}, set(), []
        expected_sources = dict(incumbent=r['champion']['source_sha256'],
                                challenger=receipt['challenger_source_sha256'], anchor=native['anchor_sha256'])
        for item in receipt.get('native_receipts', []):
            phase, side, seed = item['phase'], item['side'], item['seed']
            if phase not in losses or side not in expected_sources:
                raise ValueError('unknown paired evidence role')
            seeds = native['seeds'] if phase == 'reward' else native['validation_seeds']
            key = phase, seed, side
            if type(seed) is not int or seed not in seeds or key in seen:
                raise ValueError('duplicate or undeclared paired seed')
            seen.add(key)
            actual_order.append((phase, seed, side))
            path = (Path(output) / item['path']).resolve()
            if not path.is_relative_to(Path(output).resolve()) or sha(path) != item['sha256']:
                raise ValueError('native record outside output or changed')
            record = read_json(path)
            m, p = record['metrics'], record['provenance']
            expected_counts = dict(steps=native['budget']['steps'],
                training_tokens=native['budget']['steps'] * native['budget']['batch'] * native['budget']['context'],
                eval_tokens=native['budget']['eval_batches'] * native['budget']['batch'] * native['budget']['context'])
            if (record.get('status') != 'ok' or record.get('environment') != 'training'
                    or record.get('task') != 'tinystories-next-token' or record.get('profile') != native['profile']
                    or m.get('protocol') != native['budget'] or m.get('split') != 'dev'
                    or p.get('source_sha256') != expected_sources[side]
                    or p.get('seed') != seed or type(p.get('seed')) is not int
                    or p.get('device') != native['device'] or p.get('data_kind') != 'real'
                    or p.get('data_manifest_sha256') != native['data_manifest_sha256']
                    or m.get('evaluation_offset_tokens', 0) != (native['validation_offset_tokens'] if phase == 'validation' else 0)
                    or any(type(m.get(k)) is not int or m[k] != v for k, v in expected_counts.items())):
                raise ValueError('native paired budget/source/data/seed binding differs')
            expected_evaluator = {n: native['runtime_sha256'][n] for n in NATIVE[:7]}
            if (type(m.get('params')) is not int or not 0 < m['params'] <= native['budget']['max_params']
                    or p.get('evaluator_sha256') != expected_evaluator or p.get('suite_sha256') != native['suite_sha256']
                    or any(p.get(k) != native['runtime'][k] for k in ('python', 'torch', 'numpy', 'hardware', 'machine', 'device'))
                    or p.get('checkpoint_sha256') != sha(path.parent / 'weights.pt')
                    or p.get('initial_checkpoint_sha256') != sha(path.parent / 'initial-weights.pt')):
                raise ValueError('native runtime/checkpoint/parameter evidence differs')
            loss = m.get('held_out_loss')
            reward = record.get('reward')
            if (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                    or type(reward) not in (int, float) or not math.isfinite(reward)
                    or not math.isclose(reward, math.exp(-loss), rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError('native independent grade/reward differs')
            if (not m.get('initial_tensor_sha256') or not m.get('initial_state_spec_sha256')
                    or m.get('initial_state_spec_sha256') != m.get('final_state_spec_sha256')
                    or not m.get('final_tensor_sha256')
                    or m['final_tensor_sha256'] != m.get('loaded_final_tensor_sha256')):
                raise ValueError('native fixed-model initial/final evidence missing')
            initial = m['initial_tensor_sha256'], m['initial_state_spec_sha256']
            pair_key = phase, seed
            if pair_key in initials and initials[pair_key] != initial:
                raise ValueError('paired seeded initial model differs')
            initials[pair_key] = initial
            losses[phase][side].append(loss)
        expected = {(phase, seed, side) for phase, seeds in
                    [('reward', native['seeds']), ('validation', native['validation_seeds'])]
                    for seed in seeds for side in expected_sources}
        if seen != expected:
            raise ValueError('complete matched pair and report-only anchor evidence required')
        means = {phase: {side: math.fsum(v / len(values) for v in values)
                        for side, values in sides.items()} for phase, sides in losses.items()}
        a, b = means['reward']['incumbent'], means['reward']['challenger']
        va, vb = means['validation']['incumbent'], means['validation']['challenger']
        if (receipt.get('incumbent_metric') != a or receipt.get('challenger_metric') != b
                or receipt.get('validation', {}).get('incumbent_metric') != va
                or receipt.get('validation', {}).get('challenger_metric') != vb):
            raise ValueError('reported relative loss differs from native paired grades')
        anchor = receipt.get('anchor', {})
        if (anchor.get('report_only') is not True or anchor.get('source_sha256') != native['anchor_sha256']
                or anchor.get('reward_mean_loss') != means['reward']['anchor']
                or anchor.get('validation_mean_loss') != means['validation']['anchor']):
            raise ValueError('reported fixed anchor differs from native evidence')
        expected_orders = []
        for index, (phase, seed) in enumerate(
                [(phase, seed) for phase, seeds in [('reward', native['seeds']),
                  ('validation', native['validation_seeds'])] for seed in seeds]):
            expected_orders.append(dict(phase=phase, seed=seed, order=(['challenger', 'incumbent']
                if index % 2 else ['incumbent', 'challenger'])))
        if receipt.get('pair_orders') != expected_orders:
            raise ValueError('alternating matched pair order missing')
        expected_native_order = [(entry['phase'], entry['seed'], side)
                                 for entry in expected_orders for side in [*entry['order'], 'anchor']]
        if actual_order != expected_native_order:
            raise ValueError('native receipt sequence differs from alternating pair order')
        qualified = receipt.get('comparison_eligible') is True
        valid_win = qualified and verified_two_percent_win(va, vb)
        both_win = qualified and verified_two_percent_win(a, b) and valid_win
        if receipt.get('validation', {}).get('passed') is not valid_win or receipt.get('promotion_ready') is not both_win:
            raise ValueError('promotion must verify at least two percent on fresh validation and reward pairs')

    def evaluate(self, r, candidate_sha, receipt, output):
        # Recheck after execution/at submit as well: concurrent seed reservations
        # fail closed instead of making an old or duplicate receipt eligible.
        self.reject_reused_seeds(r['native_round'], exclude_round=r['round_id'])
        result = super().evaluate(r, candidate_sha, receipt, output)
        if candidate_sha == r['champion']['source_sha256']:
            return dict(status='invalid', score=None, promotion_eligible=False,
                        exclusion_reasons=['candidate no-op: same source as incumbent; no training reward'])
        self.verify_pair_evidence(r, receipt, output)
        current = read(self.root / 'champions' / (r['lane_id'] + '.json'))
        if current != r['champion']:
            result.update(score=None, promotion_eligible=False)
            result['exclusion_reasons'].append('stale incumbent revision; reward excluded')
        return result

    def submit(self, rid, aid):
        result = super().submit(rid, aid)
        self.export_reports()
        return result

    def reward_for_current_incumbent(self, rid, aid):
        """Revalidate one sealed result at the caller's consumption boundary.

        This returns a checked snapshot, not a lease holding the opponent fixed
        across a later SGD/group transaction. Group semantics are not implemented.
        The caller still owns exactly-once reward consumption in its trainer.
        """
        import time
        with self.locked():
            r = self.round(rid)
            self.check_sources(r)
            if not isinstance(aid, str) or len(aid) != 32 or any(c not in '0123456789abcdef' for c in aid):
                raise ValueError('invalid attempt id')
            folder = self.root / 'rounds' / rid / 'attempts' / aid
            if sha(folder / 'attempt.json') != read(folder / 'seal.json')['attempt_sha256']:
                raise ValueError('sealed attempt changed')
            attempt = read(folder / 'attempt.json')
            if sha(attempt['candidate_path']) != attempt['candidate_sha256']:
                raise ValueError('candidate source changed')
            for relative, checksum in attempt.get('proof', {}).items():
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder.resolve()) or sha(path) != checksum:
                    raise ValueError('sealed native proof changed or escaped')
            if 'receipt' in attempt:
                checked = self.evaluate(r, attempt['candidate_sha256'], attempt['receipt'], folder / 'output')
            elif attempt.get('status') in ('blocked', 'error') and attempt.get('score') is None:
                checked = dict(status=attempt['status'], score=None,
                               exclusion_reasons=attempt.get('exclusion_reasons', []))
            else:
                raise ValueError('eligible reward requires sealed native receipt')
            return dict(round_id=rid, attempt_id=aid, lane_id=r['lane_id'], status=checked['status'],
                score=checked['score'], exclusion_reasons=checked['exclusion_reasons'],
                incumbent_revision=r['champion']['revision'], incumbent_sha256=r['champion']['source_sha256'],
                candidate_sha256=attempt['candidate_sha256'], reward_policy=MOVING_POLICY,
                sealed_attempt_sha256=sha(folder / 'attempt.json'), checked_at_ns=time.time_ns(),
                group_transaction_supported=False, consumed=False)

    def export_reports(self):
        """Append-only projection of shared committed promotions, not another store."""
        with self.locked():
            history = self.root / 'history.jsonl'
            if history.exists() and history.stat().st_size > 20_000_000:
                raise ValueError('bounded history export requires an explicit archive plan')
            events = [json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
            rows, seen = [], set()
            for event in events:
                if event.get('event') not in ('submitted', 'submitted_recovered') or event.get('promoted') is not True:
                    continue
                rid, aid = event['round_id'], event['attempt_id']
                if (rid, aid) in seen:
                    continue
                r = self.moving_metadata(rid)
                if r is None:
                    continue
                folder = self.root / 'rounds' / rid / 'attempts' / aid
                if sha(folder / 'attempt.json') != read(folder / 'seal.json')['attempt_sha256']:
                    raise ValueError('promotion attempt was changed')
                attempt = read(folder / 'attempt.json')
                terminal = read(folder / 'submitted.json')
                if terminal.get('promoted') is not True or terminal.get('stale') is not False:
                    raise ValueError('projection cannot invent a promotion')
                receipt = attempt['receipt']
                if sha(attempt['candidate_path']) != attempt['candidate_sha256']:
                    raise ValueError('promotion candidate changed')
                for relative, checksum in attempt.get('proof', {}).items():
                    path = (folder / relative).resolve()
                    if not path.is_relative_to(folder.resolve()) or sha(path) != checksum:
                        raise ValueError('promotion native proof changed or escaped')
                checked = League.evaluate(r, attempt['candidate_sha256'], receipt, folder / 'output')
                self.verify_pair_evidence(r, receipt, folder / 'output')
                if (checked['promotion_eligible'] is not True or checked['score'] != attempt['score']
                        or attempt['candidate_sha256'] == r['champion']['source_sha256']):
                    raise ValueError('committed promotion lacks eligible paired evidence')
                rows.append(dict(time_ns=event['time_ns'], lane_id=r['lane_id'], round_id=rid, attempt_id=aid,
                    incumbent_source_sha256=r['champion']['source_sha256'], challenger_source_sha256=attempt['candidate_sha256'],
                    incumbent_loss=receipt['incumbent_metric'], challenger_loss=receipt['challenger_metric'],
                    validation_incumbent_loss=receipt['validation']['incumbent_metric'],
                    validation_challenger_loss=receipt['validation']['challenger_metric'],
                    anchor=receipt['anchor'], relative_reward=attempt['score'], reward_policy=MOVING_POLICY))
                seen.add((rid, aid))
            journal = self.root / 'promotions.jsonl'
            old = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            if old != rows[:len(old)]:
                raise ValueError('promotion projection differs from shared committed history')
            with journal.open('ab') as stream:
                for row in rows[len(old):]:
                    stream.write(canonical(row) + b'\n')
                stream.flush()
                import os
                os.fsync(stream.fileno())
            standings = []
            for champion in self.status()['champions']:
                if champion.get('round_id'):
                    r = self.moving_metadata(champion['round_id'])
                    if r is not None:
                        standings.append(champion)
            report = dict(schema='training-moving-incumbent-standings-v2', promotions=len(rows),
                          champions=standings, source='projection of suite.league champion CAS and sealed history',
                          anchor_is_report_only=True)
            write(self.root / 'standings.json', report)
            return report


class TrainingEpisode(Episode):
    def __init__(self, state, round_id):
        self.league, self.round_id = TrainingLeague(state), round_id

    def observation(self):
        value = super().observation()
        value['reward_version'] = MOVING_POLICY
        return value


def freeze_current(state, data, calibration, seeds, validation_seeds, *, device='cuda',
                   profile='standard', timeout_seconds=4800):
    """Server-side round construction reuses the shared champion, never a second incumbent store."""
    from protocol import get_protocol
    import uuid
    league = TrainingLeague(state)
    p = get_protocol(profile)
    anchor = HERE / 'starter.py'
    native = dict(schema_version=1, environment='training', round_id=uuid.uuid4().hex,
        workload_id='tinystories-next-token', objective_name=MOVING_OBJECTIVE, lower_is_better=True,
        reward_policy=MOVING_POLICY, min_relative_improvement=0.02, min_improvement=0,
        incumbent_source=str(anchor), incumbent_sha256=sha(anchor),
        anchor_source=str(anchor), anchor_sha256=sha(anchor), data_dir=str(Path(data).resolve()),
        data_manifest_sha256=sha(Path(data) / 'manifest.json'), profile=profile, device=device,
        seeds=list(seeds), validation_seeds=list(validation_seeds),
        validation_offset_tokens=p.eval_batches * p.batch * (p.context + 1), budget=p.record(),
        runtime=runtime_identity(device), runtime_sha256={n: sha(HERE / n) for n in NATIVE},
        suite_sha256={n: sha(HERE.parent / 'suite' / n) for n in ('records.py', 'resources.py')},
        comparison_timeout_seconds=timeout_seconds, calibration=str(Path(calibration).resolve()),
        model_identity=model_identity(anchor), oracle_sha256=sha(HERE / 'grader.py'))
    champion_path = league.root / 'champions' / (league.lane_identity(native) + '.json')
    if champion_path.exists():
        champion = read(champion_path)
        native.update(incumbent_source=champion['source_path'], incumbent_sha256=champion['source_sha256'])
    files = [HERE / n for n in NATIVE] + [HERE / 'league_adapter.py', Path(__file__),
        HERE.parent / 'suite/__init__.py', HERE.parent / 'suite/league/__init__.py',
        HERE.parent / 'suite/records.py', HERE.parent / 'suite/resources.py',
        HERE.parent / 'suite/league/controller.py', HERE.parent / 'suite/league/episode.py',
        Path(data).resolve() / 'manifest.json']
    adapter = dict(argv=[sys.executable, str(HERE / 'league_adapter.py'), '--round', '{round}',
                        '--candidate', '{candidate}', '--output', '{output}'],
                   cwd=str(HERE), trusted_files=[str(p.resolve()) for p in files])
    return league.freeze(native, adapter, max_actions=1, timeout_seconds=timeout_seconds)
