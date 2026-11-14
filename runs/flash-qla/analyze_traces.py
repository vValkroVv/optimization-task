"""Bounded GDN ownership analysis of raw Kineto update-21 rank traces.

Read-only inputs; stdlib only. Run from optimization-task. This reports captured
work and temporal overlap, never a full-update speedup or causal device wait.
"""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path

HELPER = Path('runs/bf16-expert-cache/artifacts/raw/trace-analysis/analyze.py')
HELPER_SHA = '8052d079e436a0080b4eeab4a05f0ea7b6cedd3c54087527cb1208e08f867010'
NORMALIZER = Path('runs/overnight-streaming-perf/monitorfree-perfetto-prepare.py')
NORMALIZER_SHA = '4402b99ea328542009304f7d87723b5ff46ad3cbb9c22536fda3079fe5abe597'
DEVICE = {'kernel', 'gpu_memcpy', 'gpu_memset'}
NATIVE_F = {'ChunkGatedDeltaRuleFunction', '_FusedGate'}
NATIVE_B = {n + 'Backward' for n in NATIVE_F}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(path, label, rank):
    for p, expected in ((HELPER, HELPER_SHA), (NORMALIZER, NORMALIZER_SHA)):
        if sha(p) != expected:
            raise ValueError(f'Pinned helper changed: {p}')
    spec = importlib.util.spec_from_file_location('existing_trace_metrics', HELPER)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    raw = path.read_bytes()
    events = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)['traceEvents']
    xs = [dict(e, index=i) for i, e in enumerate(events) if e.get('ph') == 'X']
    ds = [e for e in xs if e.get('cat') in DEVICE]
    hs = [e for e in xs if e.get('cat') in {'cuda_runtime', 'cuda_driver'}]
    cpus = [e for e in xs if e.get('cat') in {'cpu_op', 'user_annotation'}]
    by_corr = collections.defaultdict(list)
    for h in hs:
        c = h.get('args', {}).get('correlation')
        if c is not None:
            by_corr[c].append(h)
    links, issues = {}, []
    for d in ds:
        matches = [h for h in by_corr[d.get('args', {}).get('correlation')]
                   if h['ts'] <= d['ts'] + 1]
        if len(matches) == 1:
            links[d['index']] = matches[0]
        else:
            issues.append({'device_index': d['index'], 'matches': len(matches)})
    def contains(a, b):
        return (a.get('pid') == b.get('pid') and a['tid'] == b['tid']
                and a['ts'] <= b['ts'] and b['ts'] + b['dur'] <= a['ts'] + a['dur'] + .001)
    def owns(r, d):
        h = links.get(d['index'])
        return h is not None and r.get('pid') == h.get('pid') and r['tid'] == h['tid'] and r['ts'] <= h['ts'] < r['ts'] + r['dur']
    def family(d):
        # Fix the reused helper's EP-before-NCCL classification for DeepEP names.
        n = d['name'].lower()
        if d['cat'] == 'kernel' and any(k in n for k in ('ag_nvl', 'rs_nvl', 'deep_ep', 'deepep')):
            return 'ep'
        return metrics.fam(d)
    def summary(dev, roots, all_threads=False):
        iv = metrics.intervals(dev)
        host = {links[d['index']]['index']: links[d['index']] for d in dev}
        lo = min((r['ts'] for r in roots), default=0)
        hi = max((r['ts'] + r['dur'] for r in roots), default=lo)
        first = min((d['ts'] for d in dev), default=lo)
        last = max((d['ts'] + d['dur'] for d in dev), default=first)
        apis = [h for h in hs if any((r['ts'] <= h['ts'] and h['ts'] + h['dur'] <= r['ts'] + r['dur']) if all_threads else contains(r, h) for r in roots)]
        waits = [h for h in apis if any(s in h['name'] for s in ('Synchronize', 'WaitEvent', 'EventQuery'))]
        window = [(first, last)] if dev else []
        clip = lambda vals: [(max(first, a), min(last, b)) for a, b in vals if a < last and b > first]
        return {'calls': len(roots), 'device_events': len(dev), 'kernel_count': sum(d['cat'] == 'kernel' for d in dev),
                'kernel_sum_us': sum(d['dur'] for d in dev if d['cat'] == 'kernel'),
                'device_sum_us': sum(d['dur'] for d in dev), 'device_union_us': metrics.union(iv),
                'device_span_us': last-first, 'host_call_sum_us': sum(r['dur'] for r in roots),
                'host_call_union_us': metrics.union(metrics.intervals(roots)),
                'host_launch_sum_us': sum(h['dur'] for h in host.values()),
                'host_launch_union_us': metrics.union(metrics.intervals(list(host.values()))),
                'span_all_device_busy_us': metrics.union(clip(metrics.intervals(ds))),
                'span_all_device_gap_us': last-first-metrics.union(clip(metrics.intervals(ds))),
                'copy_overlap_us': metrics.intersect(iv, metrics.intervals([d for d in ds if d['cat'] == 'gpu_memcpy'])),
                'comm_overlap_us': metrics.intersect(iv, metrics.intervals([d for d in ds if family(d) in {'ep', 'nccl'}])),
                'host_wait_apis': {n: {'n': sum(h['name'] == n for h in waits), 'sum_us': sum(h['dur'] for h in waits if h['name'] == n)} for n in sorted({h['name'] for h in waits})},
                'device_indices': [d['index'] for d in dev]}
    wrappers = [e for e in cpus if e['name'] == 'torchtitan::gdn_chunk']
    native_f = [e for e in cpus if e['name'] in NATIVE_F]
    native_b = [e for e in cpus if e['name'] in NATIVE_B]
    vjps = [e for e in cpus if e['name'] == 'torchtitan::fla_vjp' and any(contains(e, f) for f in native_f) and any(contains(e, b) for b in native_b)]
    backward_graphs = [e for e in cpus if e['name'] == 'CompiledFunctionBackward']
    regions = []
    for f in native_f:
        v = [v for v in vjps if contains(v, f)]
        w = [w for w in wrappers if contains(w, f)]
        if len(v) + len(w) != 1:
            raise ValueError(f'Unproven GDN forward owner: {f["index"]}')
        phase = 'vjp_forward' if v else ('sac_forward' if any(contains(b, f) for b in backward_graphs) else 'original_forward')
        regions.append((phase, f))
    regions += [('native_backward', b) for b in native_b]
    regions += [('opaque_gdn', w) for w in wrappers] + [('gdn_vjp', v) for v in vjps]
    mbs = sorted([e for e in cpus if re.fullmatch(r'task.profile_update.21.microbatch.rank\d+', e['name'])], key=lambda e: e['ts'])
    rows = []
    for phase, r in regions:
        owned = [d for d in ds if owns(r, d)]
        gate = [d for d in owned if any(s in d['name'] for s in ('gdn_gate_', 'fused_gdn_gate', 'chunk_local_cumsum'))]
        rows.append({'phase': phase, 'cpu_index': r['index'], 'cpu_name': r['name'],
                     'microbatch': next((i+1 for i, m in enumerate(mbs) if m['ts'] <= r['ts'] and r['ts'] + r['dur'] <= m['ts'] + m['dur']), None),
                     'metrics': summary(owned, [r]), 'named_gate_or_cumsum': summary(gate, []),
                     'kernel_histogram': dict(collections.Counter(d['name'] for d in owned if d['cat'] == 'kernel'))})
    aggregate = {}
    for phase in sorted({phase for phase, r in regions}):
        roots = [r for p, r in regions if p == phase]
        dev = [d for d in ds if any(owns(r, d) for r in roots)]
        aggregate[phase] = summary(dev, roots)
    integrated_roots = mbs
    # Integrated window includes launch threads other than the annotation thread.
    integrated = summary([d for d in ds if d['index'] in links and any(m['ts'] <= links[d['index']]['ts'] < m['ts'] + m['dur'] for m in mbs)], integrated_roots, all_threads=True)
    integrated['selection'] = 'Selected three complete F/B microbatch launch-owned device work, across all host threads; excludes setup/prefix/tail.'
    selected = set(integrated['device_indices'])
    excluded = []
    for d in ds:
        if d['index'] in selected:
            continue
        h = links.get(d['index'])
        ancestors = sorted([r for r in cpus if h and contains(r, h)], key=lambda r: r['dur'])
        excluded.append({'device_index': d['index'], 'category': d['cat'], 'name': d['name'], 'ts_us': d['ts'], 'dur_us': d['dur'], 'host_launch': h, 'shortest_cpu_owner': ancestors[0] if ancestors else None})
    captured = summary(ds, [])
    captured['selection'] = 'All raw captured device events, including prefix/input/setup/tail; not a full accumulation update.'
    example = next((d for d in ds if 'gdn_gate_chunk_cumsum' in d['name']), None)
    return {'label': label, 'rank': rank, 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'script_sha256': sha(__file__), 'helper_sha256': HELPER_SHA, 'normalizer_sha256': NORMALIZER_SHA,
            'units': 'microseconds', 'categories': dict(collections.Counter(e.get('cat') for e in events)),
            'flow_phases': dict(collections.Counter(e.get('ph') for e in events if e.get('cat') == 'ac2g')),
            'device_census': dict(collections.Counter(d['cat'] for d in ds)), 'link_issues': issues,
            'microbatch_count': len(mbs), 'aggregate': aggregate, 'regions': rows, 'integrated_3mb': integrated, 'full_captured_device_scope': captured,
            'excluded_from_selected_3fb': excluded,
            'known_launch_example': {'device': example, 'host': links.get(example['index'])} if example else None,
            'limits': ['Aggregate device span includes intervening unrelated work; per-call rows are primary.',
                       'Named gate/cumsum is a subset; anonymous TileLang symbols are not assigned a scan substage.',
                       'Beta/contiguous/compiled gate preparation outside opaque scope is not proven by names alone.',
                       'Host gaps are timestamp diagnostics, not causal launch delay or event-specific waits.',
                       'No rank clock alignment, remaining29 microbatches, optimizer, trace speedup or numerical claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--rank', type=int, choices=[0, 1], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.label, args.rank)
    counts = {p: a['calls'] for p, a in result['aggregate'].items()}
    expected = {'original_forward': 18, 'sac_forward': 18, 'vjp_forward': 18, 'native_backward': 18, 'opaque_gdn': 36, 'gdn_vjp': 18}
    if result['microbatch_count'] != 3 or counts != expected or any(r['microbatch'] is None for r in result['regions']):
        raise ValueError(f'Unexpected three-microbatch GDN coverage: {counts}')
    if result['link_issues']:
        raise ValueError(f'Ambiguous/unmatched device launch links: {len(result["link_issues"])}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'counts': {p: a['calls'] for p, a in result['aggregate'].items()}, 'device_census': result['device_census']}))


if __name__ == '__main__':
    main()
