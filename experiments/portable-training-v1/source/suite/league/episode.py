"""Trainer API with reserved validation evidence hidden during interaction.

Operator League API retains full private receipts. This is feedback hygiene,
not secrecy from trusted local callers who can read same-user state files.
"""
from .controller import League, digest

class Episode:
    def __init__(self, state, round_id):
        self.league=League(state); self.round_id=round_id
    def _attempts(self):
        return self.league.root/'rounds'/self.round_id/'attempts'
    def _closed(self):
        return bool(list(self._attempts().glob('*/submitted.json')))
    def observation(self):
        r=self.league.round(self.round_id); native=r['native_round']
        # No reserved validation partitions/seeds/nonces, native proofs or metrics.
        public={k:native[k] for k in ('environment','workload_id','objective_name','profile','device','budget')}
        return dict(round_id=self.round_id,lane_id=r['lane_id'],round_version=digest(r),
                    incumbent_sha256=r['champion']['source_sha256'],incumbent_revision=r['champion']['revision'],
                    workload=public,max_actions=r['max_actions'],
                    actions_used=len(list(self._attempts().glob('*'))),terminal=self._closed(),
                    reward_version='league-relative-v1')
    def measure(self,source):
        if self._closed(): raise ValueError('episode already terminal')
        attempt=self.league.execute(self.round_id,source)
        # Interaction score is allowed. Native validation diagnostics and gates
        # stay operator-only so subsequent candidate edits receive no feedback.
        return {k:attempt[k] for k in ('attempt_id','round_id','status','score')}
    def submit(self,attempt_id):
        return self.league.submit(self.round_id,attempt_id)
