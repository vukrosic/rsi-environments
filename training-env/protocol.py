"""Evaluator-owned training protocol. Never import constants from a submission."""
from dataclasses import dataclass, asdict

VERSION = 'training-v1.1-fixed-steps'
VOCAB = 4096
MAX_PARAMS = 5_100_000
MAX_TENSOR_BYTES = 64_000_000

@dataclass(frozen=True)
class Protocol:
    profile: str
    steps: int
    batch: int
    context: int
    eval_batches: int
    train_seconds_limit: int
    process_timeout: int
    def record(self):
        return {'version': VERSION, **asdict(self), 'vocab': VOCAB,
                'max_params': MAX_PARAMS, 'max_tensor_bytes': MAX_TENSOR_BYTES}

def get_protocol(profile):
    if profile == 'standard':
        return Protocol(profile, 500, 32, 256, 60, 300, 600)
    if profile == 'smoke':
        return Protocol(profile, 3, 2, 16, 2, 30, 90)
    raise ValueError('profile must be smoke or standard')
