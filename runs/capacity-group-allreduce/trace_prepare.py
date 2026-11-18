"""Preserve trace bytes; create annotation-preserving display-lane derivatives."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('original', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
original = json.loads(a.original.read_bytes())
trace = json.loads(a.original.read_bytes())
events = trace['traceEvents']
tracks = collections.defaultdict(list)
categories = {'kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'}
for e in events:
    if e.get('ph') == 'X' and e.get('cat') in categories:
        tracks[e['pid'], e['tid']].append(e)
moves = {}
for (pid, tid), group in tracks.items():
    ends = []
    for e in sorted(group, key=lambda x: (x['ts'], -x['dur'])):
        lane = next((i for i, end in enumerate(ends) if end <= e['ts']), len(ends))
        if lane == len(ends):
            ends.append(0)
        ends[lane] = e['ts'] + e['dur']
        if lane:
            new_tid = 1000000 + int(tid) * 100 + lane
            if e['cat'] != 'gpu_user_annotation':
                key = (pid, tid, e['ts'])
                assert key not in moves
                moves[key] = new_tid
            e['tid'] = new_tid
for e in events:
    key = (e.get('pid'), e.get('tid'), e.get('ts'))
    if e.get('ph') == 'f' and key in moves:
        e['tid'] = moves[key]
assert len(original['traceEvents']) == len(events)
for before, after in zip(original['traceEvents'], events):
    assert {k: v for k, v in before.items() if k != 'tid'} == {k: v for k, v in after.items() if k != 'tid'}
    if before.get('tid') != after.get('tid'):
        assert before.get('ph') == 'f' or (before.get('ph') == 'X' and before.get('cat') in categories)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(trace, separators=(',', ':')))
sha = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
record = {'original': str(a.original.resolve()), 'original_sha256': sha(a.original), 'derivative': str(a.output.resolve()), 'derivative_sha256': sha(a.output), 'events': len(events), 'moved_tids': sum(x.get('tid') != y.get('tid') for x, y in zip(original['traceEvents'], events)), 'census': [{'phase': ph, 'category': cat, 'count': n} for (ph, cat), n in collections.Counter((e.get('ph'), e.get('cat')) for e in events).items()], 'transformation': 'Only overlapping device/GPU annotation display tids and matching device flow-finish tids change; every array position and all other event fields are preserved. CUDA args.stream remains unchanged.'}
a.output.with_suffix('.identity.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'census'}))
