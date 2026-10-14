"""Verifiers 0.4.0 v1 taskset + intercepted coding-policy harness."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import uuid
from typing import Literal

from openai import AsyncOpenAI
from pydantic import Field
import verifiers.v1 as vf

from .protocol import eligibility, parse_action, policy_view, record_of
from .evidence import hashes, snapshot
from .ownership import OwnedProcesses, finish_cleanup

CHECKOUT = Path(__file__).resolve().parents[3]


class CodingData(vf.TaskData):
    environment: Literal['inference', 'training', 'kernel']
    split: Literal['train', 'heldout']
    seed: int
    kernel_task: str = 'rmsnorm'


class CodingConfig(vf.TasksetConfig):
    environments: list[Literal['inference', 'training', 'kernel']] = ['inference', 'training', 'kernel']
    split: Literal['train', 'heldout'] = 'train'
    seed: int = Field(14, ge=0, lt=10000)
    count: int = Field(1, ge=1, le=100)


class CodingTask(vf.Task[CodingData]):
    @vf.reward
    async def measured_terminal(self, trace: vf.Trace) -> float:
        result = trace.info.get('rsi', {})
        # Never turn missing/blocked samples into a numeric reward.
        if not result.get('eligible'):
            trace.agent.trainable = False
            raise ValueError('excluded RSI rollout: ' + ', '.join(result.get('exclusion_reasons', ['missing terminal receipt'])))
        return result['terminal_reward']


class CodingTaskset(vf.Taskset[CodingTask, CodingConfig]):
    def load(self):
        if self.config.seed + self.config.count > 10000:
            raise ValueError('seed range exceeds frozen partition')
        for offset in range(self.config.count):
            for kind in self.config.environments:
                seed = self.config.seed + offset
                if self.config.split == 'heldout':
                    seed += 10000 if kind == 'training' else 900000000 if kind == 'kernel' else 10000
                task = 'attention' if self.config.split == 'heldout' else 'rmsnorm'
                yield CodingTask(CodingData(
                    id=f'{kind}-{self.config.split}-{seed}',
                    prompt='Generate complete Python code to improve the measured research task. '
                           'Respond with one JSON object containing exactly type (test, submit; '
                           'kernel also edit) and code (complete source). Tool observations follow each action.',
                    environment=kind, split=self.config.split, seed=seed, kernel_task=task))


class CodingHarnessConfig(vf.HarnessConfig):
    checkout: str = str(CHECKOUT)
    output_dir: str = str(CHECKOUT / 'suite/integration/evidence/episodes')
    data_dir: str = str(CHECKOUT / 'training-env/data')
    python: str = sys.executable
    device: Literal['cpu', 'mps', 'cuda'] = 'cpu'
    profile: Literal['smoke', 'standard'] = 'smoke'
    compatibility_only: bool = True
    calibration: str | None = None
    horizon: int = Field(3, ge=1, le=10)
    max_output_tokens: int = Field(8192, ge=1, le=65536)
    policy_timeout_seconds: float = Field(90, gt=0)
    tool_timeout: float = Field(700, ge=600)


class Bridge:
    def __init__(self, process, folder, log, owned=None):
        self.process, self.folder, self.log = process, folder, log
        self.owned = owned
        self._close_task = None

    @classmethod
    async def start(cls, config, data):
        folder = Path(config.output_dir).resolve() / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        source_manifest = snapshot(config.checkout, folder / 'source')
        owned = OwnedProcesses(folder)
        log = (folder / 'worker.stderr').open('wb')
        process = await asyncio.create_subprocess_exec(config.python, '-u',
            str(Path(__file__).with_name('worker.py')), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=log, start_new_session=True,
            limit=16 * 1024 * 1024, env=owned.environment())
        owned.register(process.pid)
        bridge = cls(process, folder, log, owned)
        bridge.source_manifest = source_manifest
        initial = dict(checkout=config.checkout, workspace=str(folder / 'native'),
            environment=data.environment, split=data.split, seed=data.seed,
            kernel_task=data.kernel_task, horizon=config.horizon, device=config.device,
            profile=config.profile, data_dir=config.data_dir,
            compatibility_only=config.compatibility_only, calibration=config.calibration)
        (folder / 'config.json').write_text(json.dumps(initial, indent=2))
        try:
            obs = await bridge.exchange(initial, config.tool_timeout)
            return bridge, obs
        except BaseException:
            await bridge.close()
            raise

    async def exchange(self, value, timeout):
        async with asyncio.timeout(timeout):
            self.process.stdin.write((json.dumps(value) + '\n').encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'environment process exited; inspect {self.folder}/worker.stderr')
        return json.loads(line)

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        return await finish_cleanup(self._close_task)

    async def _close(self):
        try:
            if self.owned is not None:
                await self.owned.cleanup()
            elif self.process.returncode is None:
                try: os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: await asyncio.wait_for(self.process.wait(), 3)
                except TimeoutError:
                    try: os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
            await asyncio.wait_for(self.process.wait(), 3)
        finally:
            if self.log is not None:self.log.close()


class CodingHarness(vf.Harness[CodingHarnessConfig]):
    EXECUTES_CODE = False  # model has no shell/runtime tool; trusted code goes to native evaluator
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        bridge, observation = await Bridge.start(self.config, data)
        trace.info['rsi'] = {'episode_dir': str(bridge.folder), 'eligible': False,
                             'exclusion_reasons': ['episode unfinished']}
        messages = [{'role': 'system', 'content': 'Return only the required JSON code action.'},
                    {'role': 'user', 'content': data.prompt_text}]
        transition = None
        policy_response_ids = []
        try:
            async with AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0,
                                   timeout=self.config.policy_timeout_seconds) as client:
                for turn in range(self.config.horizon):
                    messages.append({'role': 'user', 'content': json.dumps(policy_view(observation))})
                    response = await client.chat.completions.create(model=ctx.model,
                        messages=messages, max_tokens=self.config.max_output_tokens)
                    policy_response_ids.append(getattr(response, 'id', None))
                    text = response.choices[0].message.content or ''
                    messages.append({'role': 'assistant', 'content': text})
                    (bridge.folder / f'policy-{turn:02d}.txt').write_text(text)
                    try:
                        action = parse_action(text, data.environment)
                    except (ValueError, TypeError, UnicodeError) as exc:
                        # Every malformed policy turn consumes the harness budget;
                        # exhausting it is a policy failure (zero), not an infrastructure failure.
                        terminal = turn + 1 == self.config.horizon
                        transition = dict(observation={'status': 'invalid', 'reason': str(exc)},
                            reward=0.0 if terminal else None, terminated=terminal, truncated=False,
                            record={'schema_version': 1, 'environment': data.environment, 'task': data.id,
                                    'profile': self.config.profile, 'status': 'invalid', 'reward': 0.0,
                                    'metrics': {'action_parser_failure': True},
                                    'provenance': {'stage': 'policy_schema', 'measurement': 'none; policy schema rejection'}})
                    else:
                        if turn + 1 == self.config.horizon: action['type'] = 'submit'
                        (bridge.folder / f'action-{turn:02d}.py').write_text(action['code'])
                        transition = await bridge.exchange(action, self.config.tool_timeout)
                    with (bridge.folder / 'transitions.jsonl').open('a') as handle:
                        handle.write(json.dumps(transition, allow_nan=False) + '\n')
                    observation = transition['observation']
                    if transition['terminated'] or transition['truncated']: break
            eligible, reasons = eligibility(transition, split=data.split,
                compatibility_only=self.config.compatibility_only, profile=self.config.profile)
            if hasattr(bridge, 'source_manifest') and hashes(self.config.checkout) != bridge.source_manifest:
                eligible = False
                reasons.append('environment source changed during rollout')
            result = dict(episode_dir=str(bridge.folder), eligible=eligible, exclusion_reasons=reasons,
                terminal_reward=transition['reward'], terminated=transition['terminated'],
                truncated=transition['truncated'], record=record_of(transition),
                policy_response_ids=policy_response_ids,
                policy_turns=turn + 1, compatibility_only=self.config.compatibility_only)
            trace.info['rsi'] = result
            trace.agent.trainable = eligible
            (bridge.folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            return vf.ProgramResult(exit_code=0, stdout='', stderr='')
        finally:
            await bridge.close()


# An explicit Env ensures the final standing survives Agent's own trace stamping.
class CodingEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        agents.agent.trainable = task.data.split == 'train'
        await super().run(task, agents)

    async def finalize(self, task, episode):
        for trace in episode.traces:
            trace.agent.trainable = bool(trace.info.get('rsi', {}).get('eligible')) and trace.ok


def load_environment(config: vf.SingleAgentEnvConfig) -> vf.Env:
    """Strict typed loader, also available through vf.load_environment(config)."""
    if not isinstance(config, vf.SingleAgentEnvConfig):
        raise TypeError('expected resolved SingleAgentEnvConfig')
    return vf.load_environment(config)
