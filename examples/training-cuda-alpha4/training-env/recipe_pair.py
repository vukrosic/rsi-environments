"""Portable launcher for the fixed, accepted92 TinyStories CUDA LR pair.

Source preparation is stdlib-only. ML work happens only in explicit interpreter
subprocesses. This launcher neither installs dependencies nor acquires an outer
GPU lease across native commands. A completed workflow still needs independent
session review; CLI exit zero alone never means a valid training reward.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent


def read(path):
    p = Path(path)
    if p.stat().st_size > 1_000_000:
        raise ValueError('control JSON exceeds 1 MB')
    value = json.loads(p.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError('control JSON must be an object')
    return value


def save(path, value):
    with Path(path).open('x') as target:
        target.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def supervisor(source):
    spec = importlib.util.spec_from_file_location('recipe_owned_process_control', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_command


def child_environment(overlays, canonical_lock):
    env = os.environ.copy()
    inherited_lock = env.get('RESEARCH_ENVS_ACCELERATOR_LOCK')
    if inherited_lock and Path(inherited_lock).expanduser().resolve() != Path(canonical_lock).resolve():
        raise ValueError('inherited lease override differs from the canonical common lease')
    env.pop('PYTHONHOME', None)
    env.update(PYTHONPATH=os.pathsep.join(overlays), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               RESEARCH_ENVS_ACCELERATOR_LOCK=canonical_lock)
    return env


def guard_output(data, output, protected):
    if data == output or data in output.parents or output in data.parents:
        raise ValueError('output and real data directories must not overlap')
    if protected == output or protected in output.parents or output in protected.parents:
        raise ValueError('output must not overlap the frozen source checkpoint')


def prepare(args, output):
    spec_path = HERE / 'recipe_pair-spec.json'
    config = read(spec_path)
    templates = HERE / config['template_root']
    data = args.data.expanduser().resolve()
    # Preserve the explicit venv executable path; resolving its symlink can lose
    # the selected environment and silently select a different installed stack.
    python = args.python.expanduser().absolute()
    if not data.is_dir() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('existing data directory and executable --python are required')
    protected = templates.parent.resolve()
    guard_output(data, output, protected)
    overlays = [str(p.expanduser().resolve()) for p in args.overlay]
    if any(not Path(p).is_dir() for p in overlays):
        raise ValueError('every explicit overlay must be an existing directory')
    canonical_lock = str(Path.home() / '.cache/vuk-research-envs/accelerator.lock')
    env = child_environment(overlays, canonical_lock)
    root = output / 'source'
    root.mkdir()
    for name, expected in config['source_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('source pin escapes template root')
        source = templates / relative
        if not source.is_file() or source.stat().st_size > 1_000_000 or sha(source) != expected:
            raise ValueError('accepted source pin mismatch: ' + name)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(HERE / 'recipe_pair.py', root / 'recipe_pair.py')
    shutil.copyfile(spec_path, root / 'recipe_pair-spec.json')
    source_hashes = {**config['source_sha256'], 'recipe_pair.py': sha(root / 'recipe_pair.py'),
                     'recipe_pair-spec.json': sha(root / 'recipe_pair-spec.json')}
    packet = {key: config[key] for key in ('schema_version', 'recipe', 'seeds', 'validation_seeds',
              'validation_offset_tokens', 'budget', 'target_runtime', 'assets',
              'data_manifest_sha256', 'limits')}
    packet.update(version=config['version'], source_sha256=source_hashes, data_reuse=str(data),
                  lock_path=canonical_lock, state='PORTABLE_GENERATED_NOT_EXECUTED',
                  frozen_recipe='accepted92 LR0.001vs0.002, unchanged seeds/windows/budget/oracle',
                  effective_reward=None, reward_consumed=False, outer_policy_updated=False)
    py = str(python)
    packet_path, round_path = output / 'packet.json', output / 'round.json'
    calibration, comparison, controls = output / 'calibration.json', output / 'comparison', output / 'controls'
    pair = [py, str(root / 'run_pair.py'), '--packet', str(packet_path), '--round', str(round_path),
            '--controls', str(controls), '--output', str(comparison)]
    phases = [
        {'name': 'preflight', 'timeout_seconds': 30, 'argv': [py, str(root / 'prepare_round.py'),
         '--packet', str(packet_path), '--data', str(data), '--calibration', str(calibration),
         '--output', str(round_path)]},
        {'name': 'before-calibration', 'timeout_seconds': 15, 'argv': pair + ['--capture', 'before-calibration']},
        {'name': 'calibration', 'timeout_seconds': 90, 'argv': [py, str(root / 'training-env/calibrate.py'),
         '--device', 'cuda', '--idle-confirmed', '--output', str(calibration)]},
        {'name': 'after-calibration', 'timeout_seconds': 15, 'argv': pair + ['--capture', 'after-calibration']},
        {'name': 'paired-comparison', 'timeout_seconds': 900, 'argv': pair},
    ]
    for phase, seed in [('reward', 17001), ('validation', 17002)]:
        for side in ('incumbent', 'challenger'):
            run = f'{phase}-{seed}-{side}'
            phases.append({'name': 'saved-audit-' + run, 'timeout_seconds': 30,
                'argv': [py, str(root / 'training-env/verify_cuda_episode.py'),
                         '--run', str(comparison / run), '--packet', str(output / ('audit-' + side + '.json')),
                         '--training-root', str(root / 'training-env'), '--output', str(output / (run + '-verified.json'))]})
    cleanup = {'name': 'cleanup', 'timeout_seconds': 15, 'argv': pair + ['--capture', 'cleanup']}
    job = dict(schema_version=1, version=config['version'], output=str(output), python=py,
               overlays=overlays, canonical_lock=canonical_lock, preflight_only=args.preflight_only,
               operator_idle_assertion=args.idle_confirmed,
               phases=phases, cleanup=cleanup, whole_job_seconds=1200, cleanup_reserve_seconds=15,
               packet_path=str(packet_path), packet_source_sha256=source_hashes,
               wrapper_scope='Single whole-workflow watchdog plus unchanged per-phase/native supervision; cooperative code, not a hostile sandbox or hard memory cap.')
    save(output / 'job.json', job)
    packet['portable_job_sha256'] = sha(output / 'job.json')
    save(packet_path, packet)
    save(output / 'source-manifest.json', dict(version=config['version'], source_sha256=source_hashes,
         packet_sha256=sha(packet_path), job_sha256=sha(output / 'job.json'),
         asset_policy='Exact existing assets checked remotely by unchanged prepare_round; no asset copies.'))
    return job, env


def inspect_phase(name, output):
    if name == 'preflight':
        if read(output / 'preflight.json').get('status') != 'ok':
            raise RuntimeError('preflight JSON did not pass')
    elif name in ('before-calibration', 'after-calibration', 'cleanup'):
        if read(output / 'controls' / (name + '.json')).get('status') != 'ok':
            raise RuntimeError('physical/source/lease control did not pass: ' + name)
    elif name == 'calibration':
        value = read(output / 'calibration.json')
        if value.get('idle_assertion') is not True or not value.get('reference_seconds'):
            raise RuntimeError('fresh calibration JSON did not pass')
    elif name == 'paired-comparison':
        value = read(output / 'comparison/comparison.json')
        controls = read(output / 'controls/pair-control-summary.json')
        if (value.get('status') != 'ok' or value.get('comparison_eligible') is not True or
                len(value.get('native_receipts', [])) != 4 or controls.get('status') != 'ok' or
                controls.get('provisional_comparison_eligible') is not True):
            raise RuntimeError('native comparison excluded; preserve raw envelope and null effective reward')
    elif name.startswith('saved-audit-'):
        value = read(output / (name[len('saved-audit-'):] + '-verified.json'))
        if value.get('saved_checkpoint_consistency') is not True or value.get('actual_tensor_metadata_consistent') is not True:
            raise RuntimeError('saved checkpoint/device audit did not pass')


def inspect_device_policy(output, packet):
    """Compose the retained per-arm metadata without importing ML libraries."""
    policy = None
    for phase, seed in [('reward', 17001), ('validation', 17002)]:
        for side in ('incumbent', 'challenger'):
            run = output / 'comparison' / f'{phase}-{seed}-{side}'
            for role in ('worker', 'grader'):
                proof = read(run / (role + '-operator.json'))
                inv = proof['cuda_inventory']
                current = (inv.get('matmul_allow_tf32'), inv.get('cudnn_allow_tf32'))
                if (proof.get('torch_cuda_version') != packet['target_runtime']['torch_cuda'] or
                        any(type(value) is not bool for value in current)):
                    raise RuntimeError('CUDA build or explicit TF32 metadata missing')
                if policy is not None and current != policy:
                    raise RuntimeError('TF32 settings differ across native workers/graders')
                policy = current
    return {'cuda_build': packet['target_runtime']['torch_cuda'],
            'matmul_allow_tf32': policy[0], 'cudnn_allow_tf32': policy[1],
            'worker_grader_policies_equal': True}


def execute(output, deadline):
    job, packet = read(output / 'job.json'), read(output / 'packet.json')
    if sha(output / 'job.json') != packet['portable_job_sha256']:
        raise ValueError('generated job identity changed')
    root = output / 'source'
    run_command = supervisor(root / 'training-env/process_control.py')
    env = child_environment(job['overlays'], job['canonical_lock'])
    logs = output / 'supervisors'
    logs.mkdir()
    result = dict(status='blocked', effective_reward=None, reward_consumed=False,
                  outer_policy_updated=False, promoted=False, session_accepted=False,
                  native_comparison_eligible=False, preflight_only=job['preflight_only'], phases=[])

    def phase(item, cleanup=False):
        remaining = deadline - time.monotonic() - (0 if cleanup else job['cleanup_reserve_seconds'])
        if remaining <= 0:
            raise RuntimeError('whole-job deadline exhausted before ' + item['name'])
        cap = min(item['timeout_seconds'], remaining)
        name = item['name']
        receipt = {'argv': item['argv'], 'timeout_seconds': cap}
        try:
            execution = run_command(item['argv'], cwd=output, env=env, timeout=cap)
            receipt.update(returncode=execution.returncode, metadata=execution.supervisor_metadata)
            (logs / (name + '.stdout.log')).write_text(execution.stdout)
            (logs / (name + '.stderr.log')).write_text(execution.stderr)
            if (execution.returncode != 0 or execution.supervisor_metadata['status'] != 'ok' or
                    execution.supervisor_metadata['cleanup_errors'] or execution.supervisor_metadata['discovery_errors']):
                raise RuntimeError('supervised phase failed: ' + name)
            inspect_phase(name, output)  # Exit0 alone is never success.
            receipt['control_status'] = 'ok'
        except BaseException as exc:
            receipt.update(control_status='blocked', reason=type(exc).__name__ + ': ' + str(exc))
            if hasattr(exc, 'supervisor_metadata'):
                receipt['metadata'] = exc.supervisor_metadata
            raise
        finally:
            save(logs / (name + '.json'), receipt)
            result['phases'].append({'name': name, 'control_status': receipt['control_status']})

    try:
        for item in (job['phases'][:1] if job['preflight_only'] else job['phases']):
            phase(item)
        result['status'] = 'preflight_only' if job['preflight_only'] else 'completed_pending_independent_review'
        if not job['preflight_only']:
            result['device_policy'] = inspect_device_policy(output, packet)
            comparison = read(output / 'comparison/comparison.json')
            result.update(native_comparison_eligible=True,
                          native_relative_reward=comparison['relative_reward'],
                          promotion_ready=comparison.get('promotion_ready', False),
                          interaction_ce={k: comparison[k] for k in ('incumbent_metric', 'challenger_metric')},
                          reserved_validation=comparison['validation'])
    except BaseException as exc:
        result.update(status='blocked', native_comparison_eligible=False)
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if not job['preflight_only'] and (output / 'controls/before-calibration.json').exists():
            try:
                phase(job['cleanup'], cleanup=True)
                result['cleanup_control'] = 'ok'
            except BaseException as exc:
                result.update(status='blocked', native_comparison_eligible=False,
                              cleanup_control='blocked', cleanup_reason=type(exc).__name__ + ': ' + str(exc))
        else:
            result['cleanup_control'] = 'not_applicable_no_GPU_phase_started'
        save(output / 'workflow.json', result)
    return 0 if result['status'] in ('preflight_only', 'completed_pending_independent_review') else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--python', type=Path)
    ap.add_argument('--overlay', type=Path, action='append', default=[], help='Explicit existing dependency directory; repeat for multiple overlays; inherited PYTHONPATH is cleared')
    ap.add_argument('--preflight-only', action='store_true', help='Check source/runtime/assets and round only; no calibration, CUDA tensors or training')
    ap.add_argument('--idle-confirmed', action='store_true', help='Required operator assertion for a full GPU workflow; never inferred from snapshots')
    ap.add_argument('--_execute', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--_deadline', type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    output = args.output.expanduser().resolve()
    if args._execute:
        if args._deadline is None or not math.isfinite(args._deadline):
            ap.error('internal workflow requires the parent watchdog deadline')
        return execute(output, args._deadline)
    if args.data is None or args.python is None:
        ap.error('--data and --python are required')
    if not args.preflight_only and not args.idle_confirmed:
        ap.error('full GPU execution requires an operator --idle-confirmed assertion')
    started, created = time.monotonic(), False
    result = dict(status='blocked', effective_reward=None, session_accepted=False,
                  native_comparison_eligible=False, reward_consumed=False,
                  outer_policy_updated=False, promoted=False)
    try:
        config = read(HERE / 'recipe_pair-spec.json')
        guard_output(args.data.expanduser().resolve(), output,
                     (HERE / config['template_root']).parent.resolve())
        output.mkdir(parents=True, exist_ok=False)
        created = True
        job, env = prepare(args, output)
        deadline = started + job['whole_job_seconds']
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('whole-job preparation deadline exhausted')
        argv = [job['python'], str(output / 'source/recipe_pair.py'), '--_execute',
                '--output', str(output), '--_deadline', str(deadline)]
        run_command = supervisor(output / 'source/training-env/process_control.py')
        whole = {'argv': argv, 'timeout_seconds': remaining}
        try:
            execution = run_command(argv, cwd=output, env=env, timeout=remaining)
            whole.update(returncode=execution.returncode, metadata=execution.supervisor_metadata)
            (output / 'whole-workflow.stdout.log').write_text(execution.stdout)
            (output / 'whole-workflow.stderr.log').write_text(execution.stderr)
            if (execution.returncode != 0 or execution.supervisor_metadata['status'] != 'ok' or
                    execution.supervisor_metadata['cleanup_errors'] or execution.supervisor_metadata['discovery_errors']):
                raise RuntimeError('whole-workflow supervisor failed; inspect workflow and raw records')
            result.update(read(output / 'workflow.json'))
            if result['status'] not in ('preflight_only', 'completed_pending_independent_review'):
                raise RuntimeError('workflow control status is not successful')
        except BaseException as exc:
            if hasattr(exc, 'supervisor_metadata'):
                whole['metadata'] = exc.supervisor_metadata
            whole['reason'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            save(output / 'whole-job-supervisor.json', whole)
    except BaseException as exc:
        result.update(status='blocked', native_comparison_eligible=False,
                      reason=type(exc).__name__ + ': ' + str(exc))
    if not created:
        print(json.dumps(result), file=sys.stderr)
        return 2  # Never overwrite an earlier run, even its failure receipt.
    result.update(wall_seconds=time.monotonic() - started,
                  watchdog_scope='Whole supervised workflow after small source preparation; preparation elapsed is deducted from the 1200s budget, not a hard realtime filesystem bound.',
                  source_manifest='source-manifest.json' if (output / 'source-manifest.json').exists() else None,
                  verification='Workflow/native/audit receipts only; independent full-session acceptance remains separate.',
                  artifact_policy='Keep .pt arrays and full logs on the execution host; return compact controls only.')
    save(output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'effective_reward', 'session_accepted', 'native_comparison_eligible')}))
    return 0 if result['status'] in ('preflight_only', 'completed_pending_independent_review') else 2


if __name__ == '__main__':
    raise SystemExit(main())
