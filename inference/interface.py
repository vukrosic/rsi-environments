"""Minimal synchronous interface; no agent SDK or paid API required."""
from pathlib import Path
from scorer import evaluate

class InferenceEnvironment:
    def __init__(self, episode, device='cpu'):
        self.episode = Path(episode)
        self.device = device

    def step(self, solution, *, diagnostic=False):
        return evaluate(solution, self.episode, 'development', self.device, diagnostic)['record']

    def finalize(self, solution, *, diagnostic=False):
        return evaluate(solution, self.episode, 'final', self.device, diagnostic)['record']
