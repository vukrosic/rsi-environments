"""Environment side of an RL loop; an external trainer updates policy weights."""
import ast
import hashlib
import json
import secrets
import threading
from pathlib import Path
from assets import HERE
from scorer import evaluate
from shared_contract import make_record

class CodingAgentEnvironment:
    """JSON actions/observations, terminal-only rewards, explicit data partition.

    The trainer owns root/seed/split. Blocked/error transitions have reward=None
    and info.trainable=False; exclude them from RL optimization.
    """
    def __init__(self, root, *, device='cpu', max_steps=3, diagnostic=False):
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 10:
            raise ValueError('max_steps must be in [1,10]')
        if type(device) is not str or device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('device must be cpu, mps or cuda')
        if type(diagnostic) is not bool:
            raise ValueError('diagnostic must be boolean')
        self.root = Path(root).resolve()
        self.device = device
        self.max_steps = max_steps
        self.diagnostic = diagnostic
        self.done = True
        self._mutex = threading.Lock()

    def reset(self, *, split='train', seed=0):
        with self._mutex:
            return self._reset(split=split,seed=seed)

    def _reset(self, *, split='train', seed=0):
        if split not in ('train','heldout'):
            raise ValueError('split must be train or heldout')
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError('seed must be an integer')
        self.spec = json.loads((HERE / 'tasks.json').read_text())[split]
        self.split, self.seed = split, seed
        self.horizon = self.max_steps if split == 'train' else 1
        self.used = 0
        self.done = False
        self.episode = self.root / f'{split}-{seed}-{secrets.token_hex(6)}'
        self.episode.mkdir(parents=True)
        self.code = (HERE / 'solution.py').read_text()
        self.source = self.episode / 'agent.py'
        self.source.write_text(self.code)
        self._save_state()
        return self._observation()

    def _save_state(self):
        state = {'task_id': self.spec['task_id'], 'split': self.split, 'seed': self.seed,
            'steps_used': self.used, 'steps_limit': self.horizon, 'terminated': self.done,
            'source_sha256': hashlib.sha256(self.code.encode()).hexdigest()}
        (self.episode / 'rl-state.json').write_text(json.dumps(state, indent=2)+'\n')

    def _observation(self, feedback=None):
        documents = {}
        for name in ('TASK.md', 'DESIGN.md', 'RL-CONTRACT.md'):
            content = (HERE / name).read_bytes()
            documents[name] = {'content': content.decode('utf-8'),
                'sha256': hashlib.sha256(content).hexdigest()}
        return {'task_id': self.spec['task_id'], 'split': self.split,
            'evaluation_contract':'rsi.inference.smollm2-135m.v2', 'profile':'standard',
            'objective': 'Make fixed pretrained SmolLM2-135M prompt processing and token generation faster while preserving logits and greedy tokens.',
            'starter_code': self.code, 'steps_remaining': self.horizon-self.used,
            'contract_documents': documents,
            'model': json.loads((HERE/'model.json').read_text()),
            'action_schema': {'type': 'submit', 'code': 'complete self-contained Engine Python source', 'finish': 'optional boolean'},
            'constraints': {'atol': 0.002, 'rtol': 0.002, 'exact_greedy_tokens': True,
                'prompt_lengths': self.spec['lengths'], 'max_source_bytes': 65536,
                'timing_batch_size':1, 'required_cache_probe_batch_sizes':[1,2],
                'input_shape':'[batch, length]', 'decode_input_shape':'[batch, 1]',
                'output_shape':'[batch, vocab_size]',
                'worker_wall_seconds': 180, 'sampled_rss_bytes': 6_000_000_000,
                'permitted_methods': 'Any method satisfying correctness, resource and measurement contracts in TASK.md.'},
            'feedback': feedback}

    def step(self, action):
        with self._mutex:
            return self._step(action)

    def _step(self, action):
        if self.done:
            raise ValueError('reset required: episode already terminated')
        self.used += 1
        terminal = self.used >= self.horizon
        record = None
        error = None
        try:
            if not isinstance(action, dict) or action.get('type') != 'submit':
                raise ValueError('action must be {type: submit, code: ..., finish?: bool}')
            if set(action)-{'type','code','finish'}:
                raise ValueError('unknown action fields')
            if 'finish' in action and not isinstance(action['finish'], bool):
                raise ValueError('finish must be boolean')
            terminal = terminal or action.get('finish', False)
            code = action.get('code')
            if not isinstance(code, str) or len(code.encode()) > 65536:
                raise ValueError('source must be a string of at most 65536 bytes')
            (self.episode / f'action-{self.used:02d}.py').write_text(code)
            ast.parse(code)
        except (ValueError, SyntaxError, TypeError, UnicodeError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            record = make_record(environment='inference', task=self.spec['task_id'],
                profile='standard', status='invalid', reward=0.0,
                metrics={'error':error, 'correctness_pass':False, 'model_executed':False},
                provenance={'split':self.split, 'seed':self.seed,
                    'episode_dir':str(self.episode), 'stage':'action_validation'})
        else:
            # Only action validation can produce the invalid/zero label above.
            # A handled native candidate failure already arrives as an invalid
            # record. An exception escaping the evaluator is infrastructure,
            # even when its Python type is ValueError/TypeError/SyntaxError.
            try:
                self.code = code
                self.source.write_text(code)
                raw = evaluate(self.source, self.episode, 'final' if terminal else 'development',
                               self.device, self.diagnostic, partition=self.split)
                record = raw['record']
                if record['status'] in ('blocked','error'):
                    terminal = True
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                record = make_record(environment='inference', task=self.spec['task_id'],
                    profile='standard', status='error', reward=None,
                    metrics={'error':error, 'correctness_pass':None,
                        'timing_valid':False, 'model_executed':None},
                    provenance={'split':self.split, 'seed':self.seed,
                        'episode_dir':str(self.episode), 'stage':'evaluation'})
                terminal = True
        self.done = terminal
        self._save_state()
        status = record['status'] if record else 'invalid'
        trainable = status in ('ok','invalid') and self.split == 'train'
        final_reward = record['reward'] if record else 0.0
        reward = final_reward if terminal else 0.0
        observation = self._observation(record if record else {'status':'invalid','error':error})
        transition = {'observation': observation, 'reward': reward, 'terminated': terminal,
            'truncated': False, 'info': {'record': record, 'split': self.split, 'seed': self.seed,
                'trainable': trainable, 'terminal_reward': final_reward if terminal else None,
                'error': error, 'episode_dir': str(self.episode)}}
        with (self.episode / 'transitions.jsonl').open('a') as f:
            f.write(json.dumps(transition, allow_nan=False)+'\n')
        return transition
