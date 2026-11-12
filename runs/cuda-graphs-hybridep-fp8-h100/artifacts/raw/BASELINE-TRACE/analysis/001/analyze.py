#!/usr/bin/env python3
"""Read completed native Kineto traces sequentially; all times use microseconds.

This is task evidence tooling, not training instrumentation. Profile timings
describe mechanisms and never establish an unprofiled speedup.
"""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def intersect(a, b):
    a, b = union(a), union(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(intervals):
    return sum(b-a for a, b in union(intervals)) / 1000


def iv(e):
    return e['ts'], e['ts'] + e.get('dur', 0)


def family(name):
    n = name.lower()
    if any(x in n for x in ('ag_nvl', 'rs_nvl', 'hybrid_ep', 'hybridep', 'deep_ep', 'deepep')):
        return 'hybridep'
    if 'nccl' in n:
        return 'nccl'
    if any(x in n for x in ('wait_value', 'waitvalue', 'wait_kernel')):
        return 'device_wait'
    return 'compute_or_unclassified'


def native_spans(log, offset):
    stacks, result = collections.defaultdict(list), []
    for line in log.open():
        d = json.loads(line)
        name = d.get('log_type_name', '')
        if not name.endswith(('_start', '_end')):
            continue
        base, side = name.rsplit('_', 1)
        if base not in ('step', 'fwd_bwd', 'optim'):
            continue
        key = (base, d.get('step'), d.get('tid'))
        if side == 'start':
            stacks[key].append(d)
        elif stacks[key]:
            start = stacks[key].pop()
            result.append({'name': base, 'step': d['step'],
                           'ts': start['time_us']-offset,
                           'dur': d['time_us']-start['time_us']})
    return sorted(result, key=lambda x: x['ts'])


def analyze(path, log):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        raw = f.read()
    trace_hash = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    del raw
    events = [e for e in doc['traceEvents'] if e.get('ph') == 'X' and 'ts' in e]
    kernels = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e:e['ts'])
    copies = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    api = [e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    cpu = [e for e in events if e.get('cat') == 'cpu_op']
    profile = [e for e in events if e.get('name','').startswith('ProfilerStep#')]
    offset = doc.get('baseTimeNanoseconds', 0) / 1000
    spans = native_spans(log, offset)
    host_corr = {e.get('args',{}).get('correlation'):e for e in api
                 if 'Launch' in e['name'] and 'correlation' in e.get('args',{})}
    special = [e for e in cpu if any(x in e['name'].lower() for x in
               ('hybridep::','attention','chunk','float8','fp8','checkpoint','compiled','adam'))]
    syncs = [e for e in api if re.search(r'Synchroniz|EventQuery',e['name'])]
    graph = [e for e in api if re.search(r'(cuda|cu)GraphLaunch',e['name'])]
    dispatch = [e for e in cpu if e['name'] == 'hybridep::dispatch']
    hist = collections.defaultdict(lambda: [0,0.0])
    for e in kernels:
        hist[e['name']][0] += 1
        hist[e['name']][1] += e.get('dur',0)/1000
    features = {}
    patterns = {
        'hybridep_ops': r'hybridep::', 'fp8_ops': r'float8|fp8|scaled_mm|grouped_mm',
        'attention_ops': r'attention|flash', 'compiled_ops': r'Compiled|Torch-Compiled',
        'optimizer_ops': r'adam|Optimizer.step', 'loss_ops': r'cross_entropy|chunk|nll_loss',
        'sac_ops': r'checkpoint|Checkpoint',
    }
    for label, pattern in patterns.items():
        c=collections.Counter(e['name'] for e in events if re.search(pattern,e['name'],re.I))
        features[label]=dict(c.most_common(35))
    summaries=[]
    # Host/device ProfilerStep annotations duplicate the same iteration with
    # different boundaries. Use the native update's wall-clock boundaries and
    # select complete updates by dominant overlap with any profile marker.
    updates = []
    for step in (s for s in spans if s['name'] == 'step'):
        overlapping = [p for p in profile if total(intersect([iv(p)],[iv(step)])) > .9*step['dur']/1000]
        if overlapping:
            step = dict(step)
            step['markers'] = [p['name'] for p in overlapping]
            updates.append(step)
    for step in updates:
        window=[iv(step)]
        start,end=iv(step)
        gpu=intersect([iv(e) for e in kernels+copies],window)
        idle=[]; cursor=start
        for a,b in gpu:
            if a>cursor: idle.append((cursor,a))
            cursor=max(cursor,b)
        if cursor<end:idle.append((cursor,end))
        ks=[e for e in kernels if start<=e['ts']<end]
        ds=[e for e in dispatch if start<=e['ts']<end]
        contained_native=[s for s in spans if s['name']=='step' and
                          min(end,iv(s)[1])-max(start,s['ts'])>0]
        native_ids=[step['step']]
        gpu_starts=[e['ts'] for e in kernels]
        late=[]
        for a,b in idle:
            i=bisect.bisect_left(gpu_starts,b-0.01)
            if i>=len(kernels):continue
            k=kernels[i]
            h=host_corr.get(k.get('args',{}).get('correlation'))
            if h and a<h['ts']<=b:
                late.append({'gap_ms':(b-a)/1000,'host_submit_late_ms':(h['ts']-a)/1000,
                             'kernel':k['name'],'launch':h['name'],
                             'host_enclosing_special':[e['name'] for e in special
                               if e.get('tid')==h.get('tid') and e['ts']<=h['ts']<iv(e)[1]][-5:]})
        op_counts=collections.Counter(e['name'] for e in cpu if start<=e['ts']<end)
        summaries.append({'profiler_markers':step['markers'],'native_update_ids':native_ids,
            'window_ms':(end-start)/1000,'gpu_all_stream_busy_ms':total(gpu),
            'gpu_all_stream_idle_ms':total(idle),
            'gpu_family_union_ms':{f:total(intersect([iv(e) for e in ks if family(e['name'])==f],window))
                                  for f in ['hybridep','nccl','device_wait','compute_or_unclassified']},
            'gpu_copy_union_ms':total(intersect([iv(e) for e in copies],window)),
            'host_api_union_ms':total(intersect([iv(e) for e in api],window)),
            'host_launch_union_ms':total(intersect([iv(e) for e in api if 'Launch' in e['name']],window)),
            'host_sync_query_union_ms':total(intersect([iv(e) for e in syncs],window)),
            'host_sync_query_overlap_gpu_idle_ms':total(intersect([iv(e) for e in syncs],idle)),
            'dispatch_cpu_union_ms':total(intersect([iv(e) for e in ds],window)),
            'dispatch_sync_union_ms':total(intersect(intersect([iv(e) for e in syncs],[iv(e) for e in ds]),window)),
            'dispatch_sync_overlap_gpu_idle_ms':total(intersect(intersect([iv(e) for e in syncs],[iv(e) for e in ds]),idle)),
            'graph_launch_count':sum(start<=e['ts']<end for e in graph),
            'native_microbatch_counts':{str(n):sum(s['name']=='fwd_bwd' and s['step']==n for s in spans) for n in native_ids},
            'native_optimizer_counts':{str(n):sum(s['name']=='optim' and s['step']==n for s in spans) for n in native_ids},
            'hybridep_op_counts':{n:c for n,c in op_counts.items() if 'hybridep::' in n},
            'largest_host_late_gaps':sorted(late,key=lambda x:x['gap_ms'],reverse=True)[:15]})
    return {'trace':str(path),'uncompressed_sha256':trace_hash,'native_log':str(log),'trace_base_epoch_us':offset,
            'profiler_markers':[{k:e.get(k) for k in ('name','cat','ts','dur')} for e in profile],
            'event_count':len(events),'category_counts':dict(collections.Counter(e.get('cat') for e in events)),
            'graph_launch_events':[{k:e.get(k) for k in ('name','ts','dur')} for e in graph],
            'api_counts':dict(collections.Counter(e['name'] for e in api)),
            'features':features,'steps':summaries,
            'top_kernels_by_sum_ms':sorted([{'name':n,'count':v[0],'sum_ms':v[1]} for n,v in hist.items()],key=lambda x:x['sum_ms'],reverse=True)[:45]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('native',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.native.glob('profiling/traces/**/rank*_trace.json*'))
    assert len(paths)==8, f'Expected eight completed traces, got {len(paths)}'
    results=[]
    for path in paths:
        rank=int(re.search(r'rank(\d+)',path.name).group(1))
        logs=list(args.native.glob(f'structured_logs/*global_rank_{rank}.*.jsonl'))
        assert len(logs)==1,(rank,logs)
        result=analyze(path,logs[0]);results.append(result)
        (args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(f'Finished rank {rank}: {len(result["steps"])} profiler markers',flush=True)
    (args.output/'summary.json').write_text(json.dumps({'ranks':results},indent=2)+'\n')


if __name__=='__main__':main()
