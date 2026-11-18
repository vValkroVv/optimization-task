"""Color overlapping GPU display lanes; retain the original CUDA stream in args."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/schedule-embedding/artifacts/raw/trace-analysis'
GPU = {'kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'}


def prepare(source, destination):
    original = json.loads(source.read_bytes())
    derived = json.loads(source.read_bytes())
    tracks = collections.defaultdict(list)
    moves = {}
    for event in derived['traceEvents']:
        if event.get('ph') == 'X' and event.get('cat') in GPU:
            event.setdefault('args', {}).setdefault('stream', event['tid'])
            tracks[event['pid'], event['tid']].append(event)
    for (pid, tid), events in tracks.items():
        ends = []
        for event in sorted(events, key=lambda e: (e['ts'], -e['dur'])):
            lane = next((i for i, end in enumerate(ends) if end <= event['ts']), len(ends))
            if lane == len(ends):
                ends.append(0)
            ends[lane] = event['ts'] + event['dur']
            if lane:
                new = 1000000 + int(tid) * 100 + lane
                if event['cat'] != 'gpu_user_annotation':
                    key = (pid, tid, event['ts'])
                    assert key not in moves
                    moves[key] = new
                event['tid'] = new
    for event in derived['traceEvents']:
        key = (event.get('pid'), event.get('tid'), event.get('ts'))
        if event.get('ph') == 'f' and key in moves:
            event['tid'] = moves[key]
    assert len(original['traceEvents']) == len(derived['traceEvents'])
    changed_tid = added_stream = 0
    for before, after in zip(original['traceEvents'], derived['traceEvents']):
        permitted = dict(after)
        permitted['tid'] = before.get('tid')
        if before.get('ph') == 'X' and before.get('cat') in GPU:
            permitted['args'] = dict(permitted['args'])
            if 'stream' not in before.get('args', {}):
                assert permitted['args'].pop('stream') == before['tid']
                added_stream += 1
        if 'tid' not in before:
            permitted.pop('tid')
        assert before == permitted
        changed_tid += before.get('tid') != after.get('tid')
    destination.write_text(json.dumps(derived, separators=(',', ':')))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {'original': str(source.relative_to(ROOT)), 'original_sha256': digest(source),
            'derivative': str(destination.relative_to(ROOT)), 'derivative_sha256': digest(destination),
            'events': len(original['traceEvents']), 'changed_tids': changed_tid,
            'added_stream_fields': added_stream, 'verified_permitted_changes_only': True}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    records = [prepare(ROOT / f'runs/production-data-abcd/artifacts/raw/proddata-d-diagnostic-001/native/profiling/traces/iteration_21/rank{rank}_trace.json', OUT / f'd0-rank{rank}.lanes.json') for rank in (0, 1)]
    (ROOT / 'runs/schedule-embedding/trace-analysis/identities.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))
