"""Reproduce environment transitions; this does NOT train an outer agent policy."""
import argparse
import json
from pathlib import Path
from interaction import TrainingEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--profile', choices=('smoke', 'standard'), default='smoke')
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    ap.add_argument('--calibration')
    args = ap.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        ap.error('preserve existing examples; choose a new output directory')
    starter = (Path(__file__).parent / 'starter.py').read_text()
    changed = starter.replace('LR, WARMUP_FRAC, WD = 1e-3, 0.03, 0.1',
                              'LR, WARMUP_FRAC, WD = 4e-3, 0.03, 0.01')
    env = TrainingEnv(root / 'train', args.data, args.profile, args.device, 0, calibration=args.calibration)
    obs = env.reset()
    assert obs['task_split'] == 'train' and obs['terminal_reward_split'] == 'dev'
    transitions = []
    for source, final in ((starter, False), (changed, False), (changed, True)):
        result = env.step({'code': source, 'final': final})
        if result['truncated'] or result['record']['status'] != 'ok':
            raise RuntimeError(f'example failed: {result}')
        assert result['terminated'] == final and result['info']['scored_data_split'] == 'dev'
        assert (result['reward'] is not None) == final
        transitions.append(result)
    heldout = TrainingEnv(root / 'heldout', args.data, args.profile, args.device, 10000,
                          calibration=args.calibration, task_split='heldout')
    assert heldout.reset()['terminal_reward_split'] == 'final'
    result = heldout.step({'code': changed, 'final': True})
    assert result['terminated'] and result['info']['scored_data_split'] == 'final'
    if result['record']['status'] != 'ok':
        raise RuntimeError(f'heldout example failed: {result}')
    transitions.append(result)
    (root / 'transitions.json').write_text(json.dumps(transitions, indent=2) + '\n')
    summary = {'api_verified': 'training-episode-v1', 'profile': args.profile,
               'outer_policy_trained': False,
               'data_kind': result['record']['provenance']['data_kind'],
               'train_dev_losses': [t['observation']['held_out_loss'] for t in transitions[:3]],
               'train_terminal_reward': transitions[2]['reward'],
               'heldout_terminal_loss': result['observation']['held_out_loss'],
               'heldout_terminal_reward': result['reward']}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
