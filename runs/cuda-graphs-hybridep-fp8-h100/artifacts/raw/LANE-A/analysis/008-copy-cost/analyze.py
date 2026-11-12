"""Offline A008 copy inventory, sequential rank loading; no training execution."""
import bisect
import collections
import csv
import gc
import hashlib
import json
import pathlib

ROOT = pathlib.Path('runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A')
OUT = ROOT / 'analysis/008-copy-cost'
prior = json.loads((ROOT / 'analysis/008/replay-counts.json').read_text())


def union_us(events):
    total = 0.0
    end = float('-inf')
    for e in sorted(events, key=lambda e: e['ts']):
        finish = e['ts'] + e['dur']
        total += max(0, finish - max(end, e['ts']))
        end = max(end, finish)
    return total


results = []
for rank in range(8):
    path = pathlib.Path(prior[rank]['trace'])
    es = json.load(path.open())['traceEvents']
    steps = sorted((e for e in es if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('ProfilerStep#')), key=lambda e: e['ts'])
    assert [s['name'] for s in steps] == ['ProfilerStep#21', 'ProfilerStep#22', 'ProfilerStep#23']
    def step_for(e):
        return next((s['name'].split('#')[1] for s in steps if s['ts'] <= e['ts'] < s['ts'] + s['dur']), None)
    groups = {}
    for key in prior[rank]['groups']:
        comp, region = key.split(' | ')
        groups[comp] = 'mask' if '0/0' in region else 'ce_forward' if '2/0' in region else 'ce_backward'
    # CPU intervals establish actual same-thread ancestry; API correlation then
    # links to device activity. GPU timestamp proximity is not attribution.
    host = collections.defaultdict(list)
    for e in es:
        if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'):
            host[e['tid']].append(e)
    api = {}
    graph_calls = {}
    foreach = []
    for thread in host.values():
        active = []
        for e in sorted(thread, key=lambda e: (e['ts'], -e['dur'])):
            active = [p for p in active if p['ts'] + p['dur'] >= e['ts'] + e['dur']]
            parents = active
            if e.get('cat') in ('cuda_runtime', 'cuda_driver'):
                comp = next((p for p in reversed(parents) if 'Call CompiledFxGraph' in p['name']), None)
                cp = next((p for p in reversed(parents) if p['name'] == 'aten::copy_'), None)
                fc = next((p for p in reversed(parents) if p['name'] == 'aten::_foreach_copy_'), None)
                info = {'event': e, 'step': step_for(e), 'compiled': comp, 'copy': cp, 'foreach': fc,
                        'ancestors': [p['name'] for p in parents]}
                api[e.get('args', {}).get('correlation')] = info
                if e['name'] == 'cudaGraphLaunch' and comp:
                    graph_calls[id(comp)] = info
            if e['name'] == 'aten::_foreach_copy_':
                comp = next((p for p in reversed(parents) if 'Call CompiledFxGraph' in p['name']), None)
                if comp and comp['name'] in groups:
                    foreach.append((e, groups[comp['name']], step_for(e)))
            active.append(e)
    buckets = collections.defaultdict(list)
    kernels_by_corr = collections.defaultdict(list)
    for e in es:
        if e.get('cat') == 'kernel':
            kernels_by_corr[e.get('args', {}).get('correlation')].append(e)
    examples = {}
    graph_nonmemcpy = collections.Counter()
    unmatched = 0
    ordering = collections.Counter()
    for e in es:
        if e.get('cat') not in ('gpu_memcpy', 'kernel'):
            continue
        a = api.get(e.get('args', {}).get('correlation'))
        if not a or a['step'] is None:
            if e.get('cat') == 'gpu_memcpy': unmatched += 1
            continue
        comp = a['compiled']
        phase = groups.get(comp['name']) if comp else None
        replay_copy = bool(phase and a['foreach'])
        if e.get('cat') == 'kernel':
            if replay_copy:
                graph_nonmemcpy[(phase, e['name'])] += 1
            continue
        cp = a['copy']
        typ = (cp or {}).get('args', {}).get('Input type', [])
        dims = (cp or {}).get('args', {}).get('Input Dims', [])
        label = phase + '_input' if replay_copy else 'other_memcpy'
        dtype = typ[0] if typ else 'unavailable'
        shape = dims[0] if dims else None
        key = (a['step'], label, e['name'], e['args'].get('bytes'), dtype, json.dumps(shape))
        buckets[key].append((e, a))
        if replay_copy:
            launch = graph_calls.get(id(comp))
            if launch:
                ordering['host_copy_api_before_graph_launch'] += a['event']['ts'] + a['event']['dur'] <= launch['event']['ts']
                ks = kernels_by_corr[launch['event']['args']['correlation']]
                ordering['gpu_copy_before_graph_kernel_same_stream'] += bool(ks) and all(e['args']['stream'] == k['args']['stream'] and e['ts'] + e['dur'] <= k['ts'] for k in ks)
            exkey = (label, e['args'].get('bytes'), dtype)
            if exkey not in examples:
                examples[exkey] = {'device': e, 'host_api': a['event'], 'copy_op': cp, 'foreach_op': a['foreach'], 'compiled_call': comp, 'ancestors': a['ancestors'], 'graph_launch': launch['event'] if launch else None, 'replay_kernels': ks if launch else []}
    rows = []
    for key, pairs in sorted(buckets.items()):
        step, label, name, size, dtype, shape = key
        dev = [p[0] for p in pairs]
        ancestors = collections.Counter(' > '.join(p[1]['ancestors'][-5:]) for p in pairs)
        rows.append({'rank': rank, 'step': int(step), 'scope': label, 'name': name, 'bytes_each': size, 'dtype': dtype, 'shape': json.loads(shape), 'count': len(dev), 'bytes_total': sum(e['args'].get('bytes', 0) for e in dev), 'gpu_duration_sum_us': sum(e['dur'] for e in dev), 'gpu_interval_union_us': union_us(dev), 'host_api_duration_sum_us': sum(a['event']['dur'] for _, a in pairs), 'gpu_streams': sorted({e['args']['stream'] for e in dev}), 'top_cpu_ancestors': ancestors.most_common(3)})
    result = {'rank': rank, 'trace': str(path), 'rows': rows, 'ordering': dict(ordering), 'graph_input_copy_kernels': [{'phase': k[0], 'kernel': k[1], 'count': v} for k, v in graph_nonmemcpy.items()], 'unmatched_or_outside_host_steps_memcpy_count': unmatched, 'foreach': [{'phase': phase, 'step': step, 'duration_us': e['dur']} for e, phase, step in foreach], 'examples': list(examples.values())}
    results.append(result)
    (OUT / 'copy-cost.json').write_text(json.dumps(results, indent=2) + '\n')
    print(rank, 'replay copy bytes', sum(r['bytes_total'] for r in rows if r['scope'] != 'other_memcpy'), 'ordering', dict(ordering), flush=True)
    del es, host, api, buckets, result, examples, rows, foreach, graph_calls, kernels_by_corr
    gc.collect()

with (OUT / 'copy-cost.csv').open('w') as f:
    fields = ['rank', 'step', 'scope', 'name', 'bytes_each', 'dtype', 'shape', 'count', 'bytes_total', 'gpu_duration_sum_us', 'gpu_interval_union_us', 'host_api_duration_sum_us', 'gpu_streams']
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for result in results:
        writer.writerows(result['rows'])
