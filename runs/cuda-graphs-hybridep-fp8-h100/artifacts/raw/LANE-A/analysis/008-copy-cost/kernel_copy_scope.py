"""Complete copy-kernel sizes and aggregate interval unions not in memcpy table."""
import bisect
import collections
import gc
import json
import pathlib

ROOT = pathlib.Path('runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/LANE-A')
OUT = ROOT / 'analysis/008-copy-cost'
prior = json.loads((ROOT / 'analysis/008/replay-counts.json').read_text())


def union_us(events):
    end, total = float('-inf'), 0.0
    for e in sorted(events, key=lambda e: e['ts']):
        finish = e['ts'] + e['dur']
        total += max(0, finish - max(end, e['ts']))
        end = max(end, finish)
    return total


results = []
for rank in range(8):
    es = json.load(open(prior[rank]['trace']))['traceEvents']
    steps = sorted([e for e in es if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
    by_thread = collections.defaultdict(list)
    for e in es:
        if 'Call CompiledFxGraph' in e.get('name', '') and e.get('cat') == 'user_annotation':
            by_thread[e['tid']].append(e)
    for t in by_thread: by_thread[t].sort(key=lambda e: e['ts'])
    starts = {t: [e['ts'] for e in v] for t, v in by_thread.items()}
    phases = {}
    for key in prior[rank]['groups']:
        comp, region = key.split(' | ')
        phases[comp] = 'mask' if '0/0' in region else 'ce_forward' if '2/0' in region else 'ce_backward'
    def enclosing(e):
        t = e['tid']
        i = bisect.bisect_right(starts.get(t, []), e['ts']) - 1
        if i >= 0:
            p = by_thread[t][i]
            if e['ts'] + e['dur'] <= p['ts'] + p['dur']: return p
    api = {e.get('args', {}).get('correlation'): e for e in es if e.get('cat') in ('cuda_runtime', 'cuda_driver')}
    foreach = collections.defaultdict(list)
    for e in es:
        if e.get('name') == 'aten::_foreach_copy_':
            p = enclosing(e)
            if p and p['name'] in phases: foreach[id(p)].append(e)
    launches = {}
    for a in api.values():
        if a['name'] == 'cudaGraphLaunch':
            p = enclosing(a)
            if p: launches[id(p)] = a
    gpu_by_corr = collections.defaultdict(list)
    for e in es:
        if e.get('cat') == 'kernel': gpu_by_corr[e.get('args', {}).get('correlation')].append(e)
    buckets = collections.defaultdict(list)
    examples = {}
    kernel_order = collections.Counter()
    for e in es:
        if e.get('cat') not in ('kernel', 'gpu_memcpy'): continue
        a = api.get(e.get('args', {}).get('correlation'))
        if not a: continue
        step = next((s['name'].split('#')[1] for s in steps if s['ts'] <= a['ts'] < s['ts'] + s['dur']), None)
        if step is None: continue
        p = enclosing(a)
        fc = next((f for f in foreach.get(id(p), []) if f['tid'] == a['tid'] and f['ts'] <= a['ts'] and a['ts'] + a['dur'] <= f['ts'] + f['dur']), None)
        phase = phases.get(p['name']) if p else None
        if e['cat'] == 'gpu_memcpy':
            label = phase + '_input_memcpy' if phase and fc else 'other_memcpy'
        elif phase and fc:
            label = phase + '_input_kernel'
        else: continue
        buckets[(step, label)].append((e, a, fc))
        if label.endswith('_input_kernel'):
            launch = launches[id(p)]
            kernels = gpu_by_corr[launch['args']['correlation']]
            kernel_order['host_copy_api_before_graph_launch'] += a['ts'] + a['dur'] <= launch['ts']
            kernel_order['gpu_copy_before_graph_kernel_same_stream'] += bool(kernels) and all(e['args']['stream'] == k['args']['stream'] and e['ts'] + e['dur'] <= k['ts'] for k in kernels)
            examples.setdefault(label, {'device': e, 'host_api': a, 'foreach': fc, 'compiled_call': p, 'graph_launch': launch})
    rows = []
    for (step, label), pairs in sorted(buckets.items()):
        ds = [e for e, _, _ in pairs]
        row = {'rank': rank, 'step': int(step), 'scope': label, 'count': len(ds), 'gpu_duration_sum_us': sum(e['dur'] for e in ds), 'gpu_interval_union_us': union_us(ds), 'host_api_duration_sum_us': sum(a['dur'] for _, a, _ in pairs), 'gpu_streams': sorted({e['args']['stream'] for e in ds}), 'recorded_bytes_total': sum(e['args']['bytes'] for e in ds) if all('bytes' in e['args'] for e in ds) else None}
        if label.endswith('_input_kernel'):
            row['foreach_input_dims'] = list({json.dumps(fc['args'].get('Input Dims')) for _, _, fc in pairs})
            row['kernel_names'] = list({e['name'] for e in ds})
        rows.append(row)
    results.append({'rank': rank, 'rows': rows, 'kernel_order': dict(kernel_order), 'kernel_examples': examples})
    (OUT / 'copy-scopes.json').write_text(json.dumps(results, indent=2) + '\n')
    print(rank, dict(kernel_order), flush=True)
    del es, api, by_thread, foreach, gpu_by_corr, buckets, examples, launches
    gc.collect()
