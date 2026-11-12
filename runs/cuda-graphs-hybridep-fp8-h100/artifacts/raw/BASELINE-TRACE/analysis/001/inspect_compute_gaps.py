#!/usr/bin/env python3
"""Inspect representative compute-gap dependencies in a completed rank trace."""
import argparse
import collections
import json
from pathlib import Path
from analyze import family, intersect, iv, native_spans, total, union


ap=argparse.ArgumentParser();ap.add_argument('native',type=Path);ap.add_argument('output',type=Path)
a=ap.parse_args()
p=next(a.native.glob('profiling/traces/**/rank0_trace.json'))
d=json.loads(p.read_bytes())
ev=[e for e in d['traceEvents'] if e.get('ph')=='X' and 'ts' in e]
cpu=[e for e in ev if e.get('cat')=='cpu_op']
api=[e for e in ev if e.get('cat') in ('cuda_runtime','cuda_driver')]
kc=[e for e in ev if e.get('cat')=='kernel']
compute=sorted([e for e in kc if family(e['name'])=='compute_or_unclassified'],key=lambda e:e['ts'])
comm=[iv(e) for e in kc if family(e['name']) in ('hybridep','nccl')]
launch={e.get('args',{}).get('correlation'):e for e in api if 'Launch' in e['name']}
sync=[e for e in api if 'Synchroniz' in e['name']]
spans=native_spans(next(a.native.glob('structured_logs/*global_rank_0.*.jsonl')),d['baseTimeNanoseconds']/1000)
window=[iv(s) for s in spans if s['name']=='step' and s['step']==22]
busy=intersect([iv(e) for e in compute],window)
gaps=[(busy[i][1],busy[i+1][0]) for i in range(len(busy)-1) if busy[i+1][0]>busy[i][1]]
def enclosing(h):
    return sorted([e for e in cpu if e.get('tid')==h.get('tid') and e['ts']<=h['ts']<iv(e)[1]],key=lambda e:e.get('dur',0))
out=[]
for lo,hi in sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:35]:
    nxt=next(e for e in compute if abs(e['ts']-hi)<.01)
    host=launch.get(nxt.get('args',{}).get('correlation'))
    sy=[e for e in sync if total(intersect([iv(e)],[(lo,hi)]))>0]
    out.append({'start_ts':lo,'end_ts':hi,'gap_ms':(hi-lo)/1000,
        'communication_overlap_ms':total(intersect(comm,[(lo,hi)])),
        'host_sync_overlap_ms':total(intersect([iv(e) for e in sy],[(lo,hi)])),
        'next_kernel':nxt['name'],'next_stream':nxt.get('args',{}).get('stream'),
        'next_launch':None if host is None else {k:host.get(k) for k in ('name','ts','dur')},
        'next_launch_after_gap_start_ms':None if host is None else (host['ts']-lo)/1000,
        'next_launch_enclosing_ops':[] if host is None else [e['name'] for e in enclosing(host)][:12],
        'syncs':[{'name':e['name'],'dur_ms':e['dur']/1000,'enclosing_ops':[x['name'] for x in enclosing(e)][:8]} for e in sy]})
coverage=[]
for n in (22,23,24):
    ss=[s for s in spans if s['step']==n and s['name'] in ('fwd_bwd','optim')]
    coverage.append({'native_update':n,'spans':ss})
result={'trace':str(p),'native_update':22,'timer':'union gaps between non-communication kernels across all compute-bearing streams; communication can mask them',
    'compute_stream_counts':dict(collections.Counter(e.get('args',{}).get('stream') for e in compute)),
    'compute_busy_union_ms':total(busy),'compute_gaps_union_ms':total(gaps),
    'compute_gap_comm_overlap_ms':total(intersect(gaps,comm)),
    'largest_compute_gaps':out,'native_coverage':coverage}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('largest_compute_gaps','native_coverage')}))
