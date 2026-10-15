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
    diagnostic: str | None = None
    def record(self):
        fields = asdict(self)
        if self.diagnostic is None:
            fields.pop('diagnostic')
        return {'version': 'training-diagnostic-cpu-v1' if self.diagnostic else VERSION, **fields, 'vocab': VOCAB,
                'max_params': MAX_PARAMS, 'max_tensor_bytes': MAX_TENSOR_BYTES}

def get_protocol(profile, diagnostic=None):
    if diagnostic is not None:
        if profile != "smoke" or diagnostic != "cpu-training-v1":
            raise ValueError("diagnostic cpu-training-v1 is allowed only with profile smoke")
        return Protocol(profile, 20, 4, 64, 4, 90, 150, diagnostic)
    if profile == 'standard':
        return Protocol(profile, 500, 32, 256, 60, 300, 600)
    if profile == 'smoke':
        return Protocol(profile, 3, 2, 16, 2, 30, 90)
    raise ValueError('profile must be smoke or standard')
