"""Read-only, report-only exact token passage audit; never a training gate."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import math
import json
from pathlib import Path
import time
import numpy as np
from data import FILES, load_data


def window_digest(tokens):
    """Canonical integer representation permits comparison across integer dtypes."""
    return hashlib.sha256(np.asarray(tokens, dtype='<u8').tobytes()).digest()[:16]


def array_hash(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), 131072):
        digest.update(array[start:start + 131072].tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(root, profile='standard', window=64, stride=32, max_windows=1000000, timeout_seconds=45):
    if any(type(value) is not int for value in (window, stride, max_windows)) or window < 1 or stride < 1 or not 1 <= max_windows <= 1000000:
        raise ValueError('positive window/stride and max_windows between 1 and 1000000 required')
    if timeout_seconds is not None and (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('timeout must be positive or None')
    started = time.monotonic()
    def expired():
        return timeout_seconds is not None and time.monotonic() - started >= timeout_seconds
    source_before = file_hash(Path(__file__))
    root = Path(root)
    *arrays, manifest = load_data(root, profile=profile, use_cache=False)
    names = list(FILES)
    before = [array_hash(a) for a in arrays]
    provenance = {'arrays': {s: {'sha256': h, 'dtype': str(a.dtype), 'tokens': len(a)} for s, a, h in zip(names, arrays, before)},
                  'files': {p.name: file_hash(p) for p in [root / f for f in FILES.values()] + [root / 'manifest.json', root / 'tok.json'] if p.exists()}}
    eot_id = manifest['validation'].get('eot_id', 0)
    def story_ends(array):
        for offset in range(0, len(array), 65536):
            for relative in np.flatnonzero(array[offset:offset+65536] == eot_id):
                yield offset + int(relative)
    records = np.empty(max_windows, dtype=[('digest', 'V16'), ('split', 'u1'), ('start', '<u8'), ('story', '<u8')])
    scanned = {s: 0 for s in names}
    short = {s: 0 for s in names}
    count = 0
    reason = None
    for split, (name, array) in enumerate(zip(names, arrays)):
        story_start = 0
        story_number = 0
        # Synthetic streams may lack EOT: audit only completed EOT-delimited segments.
        for end in story_ends(array):
            if expired():
                reason = 'timeout'; break
            if end - story_start < window:
                short[name] += 1
            for start in range(story_start, end - window + 1, stride):
                if count == max_windows or expired():
                    reason = 'max_windows' if count == max_windows else 'timeout'; break
                records[count] = (window_digest(array[start:start + window]), split, start, story_number)
                count += 1
                scanned[name] += 1
            if reason: break
            story_start = end + 1
            story_number += 1
        if reason: break
    records = records[:count]
    records.sort(order='digest', kind='quicksort')
    pairs = {f'{names[a]}:{names[b]}': {'shared_unique_windows': 0, 'occurrences': {names[a]: 0, names[b]: 0}, 'examples': []} for a,b in itertools.combinations(range(3), 2)}
    checked = 0
    i = 0
    while i < count:
        if expired(): reason = 'timeout'; break
        j = i + 1
        while j < count and records[j]['digest'] == records[i]['digest']:
            j += 1
            if expired(): reason = 'timeout'; break
        if expired(): break
        if j - i == 1:
            checked += 1
            i = j
            continue
        if j == i + 1:
            checked += 1
            i = j
            continue  # a singleton has no cross-split pair to compare
        # Every bucket is partitioned by actual array equality, including collisions.
        groups = np.zeros(j-i, dtype=[('rep','u8'), ('counts','u8',(3,)), ('locations','u8',(3,))])
        group_count = 0
        for k in range(i, j):
            if expired(): reason = 'timeout'; break
            row = records[k]
            tokens = arrays[int(row['split'])][int(row['start']):int(row['start']) + window]
            group_id = None
            for g in range(group_count):
                if expired():
                    reason = 'timeout'; break
                rr = records[int(groups[g]['rep'])]
                if np.array_equal(tokens, arrays[int(rr['split'])][int(rr['start']):int(rr['start']) + window]):
                    group_id = g; break
            if group_id is None:
                group_id = group_count
                groups[group_id]['rep'] = k
                group_count += 1
            group = groups[group_id]
            sid = int(row['split'])
            group['counts'][sid] += 1
            group['locations'][sid] = k
        if expired(): break
        for group in groups[:group_count]:
            for a,b in itertools.combinations(range(3),2):
                if group['counts'][a] and group['counts'][b]:
                    pair = pairs[f'{names[a]}:{names[b]}']
                    pair['shared_unique_windows'] += 1
                    for sid in (a,b): pair['occurrences'][names[sid]] += int(group['counts'][sid])
                    if len(pair['examples']) < 10:
                        pair['examples'].append({names[sid]: {'story_index': int(records[int(group['locations'][sid])]['story']), 'token_offset': int(records[int(group['locations'][sid])]['start'])} for sid in (a,b)})
        checked += j-i
        i = j
    after = [array_hash(a) for a in arrays]
    files_after = {name: file_hash(root / name) for name in provenance['files']}
    assets_unchanged = provenance['files'] == files_after
    if not assets_unchanged or before != after:
        reason = 'assets_changed_during_audit'
    return {'mode': 'REPORT_ONLY', 'audit_source_sha256': source_before, 'profile': profile, 'data_kind': manifest['kind'],
            'synthetic': manifest['kind'] == 'synthetic', 'status': 'invalid_assets' if reason == 'assets_changed_during_audit' else 'partial' if reason else 'complete_sampled_scan',
            'stop_reason': reason, 'window': window, 'stride': stride, 'max_windows': max_windows,
            'timeout_seconds': timeout_seconds, 'elapsed_seconds': time.monotonic()-started,
            'scanned_windows': scanned, 'processed_bucket_windows': checked, 'skipped_short_stories': short,
            'split_pairs': pairs, 'provenance': provenance, 'arrays_unchanged': before == after, 'assets_unchanged': assets_unchanged, 'files_after': files_after, 'eot_id': eot_id,
            'count_semantics': 'Shared unique windows count distinct exact token sequences per split pair. Occurrences count sampled locations of those shared sequences in each split; not Cartesian occurrence pairs. Short stories consume zero window budget. Partial results are lower bounds from fully verified digest buckets.',
            'limitations': ['Stride sampling can miss shifted passages, including shifts not divisible by stride.', 'Only windows within completed EOT-delimited stories are scanned; no boundary crossing or incomplete tail.', 'Exact token matches cannot detect paraphrases or establish source authenticity.', 'Synthetic smoke data is not real-corpus evidence.', 'Wall timeout includes validation and provenance but cannot interrupt the loader, hashing or NumPy sort; these may exceed it.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--profile', choices=['standard','smoke'], default='standard')
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--max-windows', type=int, default=1000000)
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists(): parser.error('refusing to overwrite output')
    report = audit(args.data, args.profile, args.window, args.stride, args.max_windows, args.timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')

if __name__ == '__main__': main()
