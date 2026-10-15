"""Environment-side coding-agent RL episode API, without an outer policy trainer."""
import argparse
import fcntl
import json
import math
import time
import uuid
from pathlib import Path
from evaluate import evaluate
from protocol import get_protocol
from suite.records import make_record, write_record

EPISODE_API_VERSION = 'training-episode-v1'


class SessionBusy(TimeoutError):
    """The session owner prevented bounded cooperative lock acquisition."""


class TrainingEnv:
    """Two-level environment: candidate trains a small LM; trainer learns agent policy.

    An action is {'code': source_text, 'final': bool}; a source path remains
    available for human CLI use. step returns observation/reward/terminated/
    truncated/info, plus the full measured record. Reward at terminal train
    episodes uses dev; heldout episodes use final. The external trainer must
    not optimize on heldout episodes. Constructor and reset may raise
    SessionBusy before an action. A busy step/final truncates only the requesting
    transition; unlike other infrastructure failures it does not seal the live
    owner or change its state. lock_timeout_seconds is operational, not persisted.
    """
    def __init__(self, session_dir, data_dir, profile='smoke', device='cpu', seed=0,
                 max_trials=10, calibration=None, task_split='train', lock_timeout_seconds=2):
        if (isinstance(lock_timeout_seconds, bool)
                or not isinstance(lock_timeout_seconds, (int, float))
                or not math.isfinite(lock_timeout_seconds)
                or not 0 <= lock_timeout_seconds <= 60):
            raise ValueError('lock_timeout_seconds must be finite and in [0,60]')
        self.lock_timeout_seconds = lock_timeout_seconds
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or not 1 <= max_trials <= 10:
            raise ValueError('max_trials must be an integer in [1,10]')
        if task_split not in {'train', 'heldout'}:
            raise ValueError('task_split must be train or heldout')
        lower = 0 if task_split == 'train' else 10_000
        if isinstance(seed, bool) or not isinstance(seed, int) or not lower <= seed < lower + 10_000:
            raise ValueError(f'{task_split} episode seed must be in [{lower},{lower + 9999}]')
        get_protocol(profile)
        self.root = Path(session_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'session.json'
        self.config = {'api_version': EPISODE_API_VERSION, 'data': str(Path(data_dir).resolve()),
                       'profile': profile, 'device': device, 'seed': seed,
                       'task_split': task_split, 'max_trials': max_trials,
                       'calibration': str(Path(calibration).resolve()) if calibration else None}
        with self._lock():
            if self.state_path.exists():
                if self._read()['config'] != self.config:
                    raise ValueError('existing session has different configuration; preserve it and create a new episode')
            else:
                self._write({'config': self.config, 'trials': 0, 'finalized': False})

    def _lock(self):
        class Lease:
            def __enter__(inner):
                inner.handle = (self.root / '.session.lock').open('a+')
                deadline = time.monotonic() + self.lock_timeout_seconds
                try:
                    while True:
                        try:
                            fcntl.flock(inner.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            return inner
                        except BlockingIOError:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise SessionBusy('blocked infrastructure: session lock is busy; '
                                                  'live session state is unchanged')
                            time.sleep(min(.02, remaining))
                except BaseException:
                    inner.handle.close()
                    raise
            def __exit__(inner, *args):
                inner.handle.close()
        return Lease()

    def _read(self):
        state = json.loads(self.state_path.read_text())
        if not isinstance(state, dict) or state.get('config') != self.config:
            raise ValueError('session state/config is corrupt or changed')
        trials = state.get('trials')
        if type(trials) is not int or not 0 <= trials <= self.config['max_trials']:
            raise ValueError('session trial count is corrupt or outside its fixed budget')
        if type(state.get('finalized')) is not bool:
            raise ValueError('session finalized flag must be boolean')
        return state

    def _write(self, state):
        temp = self.root / 'session.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(self.state_path)

    def reset(self):
        """Return task state without erasing history; raises SessionBusy on contention."""
        with self._lock():
            state = self._read()
            return {'api_version': EPISODE_API_VERSION,
                    'task_id': f'tinystories-{self.config["task_split"]}-{self.config["seed"]}',
                    'task_split': self.config['task_split'], 'task': 'improve TinyStories next-token held-out loss',
                    'candidate_api': ['build_model()', 'build_optimizer(model)', 'train_step(model,opt,x,y,frac)'],
                    'action_schema': {'code': 'complete Python source (UTF-8, at most 256 KB)', 'final': 'boolean'},
                    'protocol': get_protocol(self.config['profile']).record(),
                    'profile': self.config['profile'], 'trials_remaining': self.config['max_trials'] - state['trials'],
                    'terminal_reward_split': 'dev' if self.config['task_split'] == 'train' else 'final',
                    'finalized': state['finalized'],
                    'starter_code': (Path(__file__).parent / 'starter.py').read_text(),
                    'editable_starter': str(Path(__file__).parent / 'starter.py'),
                    'last_feedback': state.get('last_feedback')}

    def step(self, action):
        if isinstance(action, dict):
            if set(action) - {'code', 'final'} or not isinstance(action.get('code'), str):
                raise ValueError('action must contain code:string and optional final:boolean only')
            if type(action.get('final', False)) is not bool:
                raise ValueError('final must be boolean')
            if not action['code'].strip() or len(action['code'].encode('utf-8')) > 256_000:
                raise ValueError('candidate code must be nonempty and at most 256 KB')
            return self._submit(action['code'], final=action.get('final', False), source_text=True)
        if not isinstance(action, (str, Path)):
            raise ValueError('action must be a source path or a code action dictionary')
        return self._submit(action, final=False)

    def final(self, candidate):
        """Human convenience API; policy agents use step({'code': ..., 'final': True})."""
        return self._submit(candidate, final=True)

    def _submit(self, candidate, final, source_text=False):
        try:
            return self._submit_locked(candidate, final, source_text)
        except SessionBusy as error:
            c = self.config
            split = 'final' if final and c['task_split'] == 'heldout' else 'dev'
            record = make_record(environment='training', task='tinystories-next-token',
                                 profile=c['profile'], status='blocked', reward=None,
                                 metrics={'reason': str(error), 'operational_phase': 'lockbusy',
                                          'session_state_unchanged': True},
                                 provenance={'api_version': EPISODE_API_VERSION,
                                             'task_split': c['task_split'], 'seed': c['seed'],
                                             'lock_timeout_seconds': self.lock_timeout_seconds})
            receipt = self.root / 'busy-receipts' / f'{uuid.uuid4()}.json'
            write_record(receipt, record)
            feedback = {'status': 'blocked', 'held_out_loss': None, 'reason': str(error),
                        'measured_dev_reward': None, 'trials_remaining': None,
                        'requires_terminal_action': False}
            return {'observation': feedback, 'reward': None, 'terminated': final,
                    'truncated': True,
                    'info': {'api_version': EPISODE_API_VERSION, 'task_split': c['task_split'],
                             'record_path': str(receipt), 'trainable': False,
                             'reward_eligibility_reasons': ['infrastructure failure'],
                             'exclude_from_rl': True, 'scored_data_split': split,
                             'operational_phase': 'lockbusy', 'session_state_unchanged': True},
                    'record': record}

    def _submit_locked(self, candidate, final, source_text=False):
        with self._lock():
            state = self._read()
            if state['finalized']:
                raise RuntimeError('session already finalized')
            if not final and state['trials'] >= self.config['max_trials']:
                raise RuntimeError('dev trial budget exhausted; submit terminal action')
            if final:
                state['finalized'] = True
                name = 'final'
            else:
                state['trials'] += 1
                name = f'dev-{state["trials"]:02d}'
            # Crash-safe reservation. Failed attempts consume the same fixed budget.
            self._write(state)
            if source_text:
                actions = self.root / 'actions'
                actions.mkdir(exist_ok=True)
                path = actions / f'{name}.py'
                path.write_text(candidate)
                candidate = path
            c = self.config
            split = 'final' if final and c['task_split'] == 'heldout' else 'dev'
            record = evaluate(candidate, c['data'], self.root / name, c['profile'],
                              split, c['device'], c['seed'], c['calibration'])
            # Intermediate score is tool feedback. A terminal-only RL reward
            # avoids making repeated dev tries themselves an extra reward source.
            truncated = record['status'] in {'blocked', 'error'}
            if truncated:
                state['finalized'] = True
                self._write(state)
            terminal_reward = record['reward'] if final else None
            feedback = {'status': record['status'], 'held_out_loss': record['metrics'].get('held_out_loss'),
                        'reason': record['metrics'].get('reason'), 'measured_dev_reward': record['reward'] if not final else None,
                        'trials_remaining': c['max_trials'] - state['trials'],
                        'requires_terminal_action': not final and not truncated and state['trials'] >= c['max_trials']}
            state['last_feedback'] = feedback
            self._write(state)
            eligibility_reasons = []
            if not final: eligibility_reasons.append('nonterminal')
            if c['task_split'] != 'train': eligibility_reasons.append('heldout evaluation only')
            if c['profile'] != 'standard': eligibility_reasons.append('smoke compatibility only')
            if record['status'] in {'blocked', 'error'}: eligibility_reasons.append('infrastructure failure')
            return {'observation': feedback, 'reward': terminal_reward, 'terminated': final,
                    'truncated': truncated,
                    'info': {'api_version': EPISODE_API_VERSION, 'task_split': c['task_split'],
                             'record_path': str(self.root / name / 'record.json'),
                             'trainable': not eligibility_reasons,
                             'reward_eligibility_reasons': eligibility_reasons,
                             'exclude_from_rl': record['status'] in {'blocked', 'error'},
                             'scored_data_split': split}, 'record': record}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('reset', 'step', 'final'))
    ap.add_argument('--session', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--candidate')
    ap.add_argument('--profile', choices=('smoke', 'standard'), default='standard')
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task-split', choices=('train', 'heldout'), default='train')
    ap.add_argument('--max-trials', type=int, default=10)
    ap.add_argument('--calibration')
    args = ap.parse_args()
    env = TrainingEnv(args.session, args.data, args.profile, args.device, args.seed, args.max_trials,
                      args.calibration, args.task_split)
    if args.action != 'reset' and not args.candidate:
        ap.error('--candidate required for step/final')
    result = env.reset() if args.action == 'reset' else getattr(env, args.action)(args.candidate)
    print(json.dumps(result, allow_nan=False))

if __name__ == '__main__':
    main()
