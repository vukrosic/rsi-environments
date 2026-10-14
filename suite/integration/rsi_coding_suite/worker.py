"""One native environment per process; avoids colliding top-level module names.

This is import/process separation for trusted local code, NOT a security sandbox.
"""
import contextlib
import json
from pathlib import Path
import sys


def main():
    config = json.loads(sys.stdin.readline())
    root = Path(config['checkout']).resolve()
    workspace = Path(config['workspace']).resolve()
    kind = config['environment']
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(root))
        if kind == 'inference':
            sys.path.insert(0, str(root / 'inference'))
            from rl_environment import CodingAgentEnvironment
            env = CodingAgentEnvironment(workspace, device=config['device'],
                max_steps=config['horizon'], diagnostic=config['compatibility_only'])
            observation = env.reset(split=config['split'], seed=config['seed'])
        elif kind == 'training':
            sys.path.insert(0, str(root / 'training-env'))
            from interaction import TrainingEnv
            env = TrainingEnv(workspace, config['data_dir'], profile=config['profile'],
                device=config['device'], seed=config['seed'], task_split=config['split'],
                max_trials=max(1, config['horizon'] - 1), calibration=config['calibration'])
            observation = env.reset()
        elif kind == 'kernel':
            sys.path.insert(0, str(root / 'kernel-env'))
            from kernel_env.environment import KernelEnvironment
            env = KernelEnvironment(config['kernel_task'], workspace,
                split='train' if config['split'] == 'train' else 'eval',
                device=config['device'], profile=config['profile'], seed=config['seed'],
                max_steps=config['horizon'], calibration=config['calibration'])
            observation = env.reset()
        else:
            raise ValueError('unknown environment')
    print(json.dumps(observation, allow_nan=False), flush=True)
    for line in sys.stdin:
        action = json.loads(line)
        if kind == 'inference':
            action = {'type': 'submit', 'code': action['code'], 'finish': action['type'] == 'submit'}
        elif kind == 'training':
            action = {'code': action['code'], 'final': action['type'] == 'submit'}
        with contextlib.redirect_stdout(sys.stderr):
            transition = env.step(action)
        print(json.dumps(transition, allow_nan=False), flush=True)
        if transition['terminated'] or transition['truncated']:
            break


if __name__ == '__main__':
    main()
