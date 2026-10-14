"""Bounded policy action decoding and conservative reward eligibility."""
import hashlib
import json
import math

MAX_SOURCE_BYTES = {'inference': 65536, 'training': 256000, 'kernel': 2097152}


def parse_action(text, environment):
    limit = MAX_SOURCE_BYTES[environment]
    if not isinstance(text, str) or len(text.encode()) > limit * 6 + 1024:
        raise ValueError('action text missing or too large')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    action = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(action, dict) or set(action) != {'type', 'code'}:
        raise ValueError('exactly type and code are required')
    allowed = {'test', 'submit', 'edit'} if environment == 'kernel' else {'test', 'submit'}
    if action['type'] not in allowed:
        raise ValueError('unsupported action type')
    if not isinstance(action['code'], str) or not action['code'].strip():
        raise ValueError('code must be nonempty source')
    if len(action['code'].encode()) > limit:
        raise ValueError('source exceeds environment limit')
    return action


def record_of(transition):
    info = transition.get('info', {})
    return transition.get('record') or info.get('record') or (info if 'status' in info else {})


def eligibility(transition, *, split, compatibility_only, profile):
    reward = transition.get('reward')
    record = record_of(transition)
    info = transition.get('info', {})
    reasons = []
    if split != 'train': reasons.append('evaluation-only partition')
    if compatibility_only or profile != 'standard': reasons.append('compatibility/smoke measurement')
    if not transition.get('terminated') or transition.get('truncated'): reasons.append('nonterminal or truncated')
    if record.get('status') not in {'ok', 'invalid'}: reasons.append('missing/blocked/error record')
    if info.get('trainable') is False or info.get('exclude_from_rl'): reasons.append('native exclusion')
    provenance = record.get('provenance', {})
    if info.get('split', info.get('task_split', 'train')) != 'train' or provenance.get('partition', 'train') != 'train':
        reasons.append('native evaluation partition')
    parser_rejection = record.get('provenance', {}).get('stage') == 'policy_schema'
    if not parser_rejection and record.get('environment') == 'training' and info.get('scored_data_split') != 'dev':
        reasons.append('training terminal must score dev partition')
    if not parser_rejection and record.get('environment') == 'kernel' and record.get('task') not in {'rmsnorm','softmax','swiglu','cross_entropy'}:
        reasons.append('evaluation-only kernel task')
    if record.get('profile', profile) != 'standard': reasons.append('native smoke profile')
    metrics = record.get('metrics', {})
    if record.get('status') == 'ok' and (metrics.get('timing_valid') is False or metrics.get('timing_reliable') is False):
        reasons.append('unvalidated timing')
    if record.get('reward') != reward: reasons.append('terminal and record rewards disagree')
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        reasons.append('null or invalid reward')
    return not reasons, reasons


def policy_view(value):
    # Operator paths/provenance and scorer internals are never policy context.
    private = {'candidate', 'editable_starter', 'record_path', 'episode_dir',
               'provenance', 'evaluator_hashes', 'scorer_sha256', 'source_snapshot',
               'artifact_dir', 'workers', 'conditions', 'machine_before', 'machine_after'}
    if isinstance(value, dict):
        return {k: policy_view(v) for k, v in value.items() if k not in private}
    if isinstance(value, list): return [policy_view(v) for v in value]
    return value


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()
